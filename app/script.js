(function () {
    const locale = navigator.language || 'en-US';

    function compareIgnoreCase(a, b) {
        const valueA = (a ?? '').toString();
        const valueB = (b ?? '').toString();
        const lowerA = valueA.toLocaleLowerCase(locale);
        const lowerB = valueB.toLocaleLowerCase(locale);
        if (lowerA < lowerB) {
            return -1;
        }
        if (lowerA > lowerB) {
            return 1;
        }
        if (valueA < valueB) {
            return -1;
        }
        if (valueA > valueB) {
            return 1;
        }
        return 0;
    }

    function sortedIgnoreCase(values) {
        if (!values) {
            return [];
        }
        if (values instanceof Set) {
            return Array.from(values).map(value => (value ?? '').toString()).sort(compareIgnoreCase);
        }
        return Array.from(values).map(value => (value ?? '').toString()).sort(compareIgnoreCase);
    }

    function ensureArray(value) {
        if (!value) {
            return [];
        }
        if (Array.isArray(value)) {
            return value;
        }
        if (value instanceof Set) {
            return Array.from(value);
        }
        if (typeof value === 'object') {
            return Object.values(value);
        }
        return [value];
    }

    function normalizeRelations(rawRelations) {
        return ensureArray(rawRelations)
            .map(entry => {
                if (!entry) {
                    return null;
                }
                if (Array.isArray(entry)) {
                    const [subject, predicate, object] = entry;
                    if (
                        subject === undefined ||
                        predicate === undefined ||
                        object === undefined
                    ) {
                        return null;
                    }
                    return [subject.toString(), predicate.toString(), object.toString()];
                }
                if (typeof entry === 'object') {
                    const subject = entry.subject ?? entry.source;
                    const predicate = entry.predicate ?? entry.relation ?? entry.label;
                    const object = entry.object ?? entry.target;
                    if (
                        subject === undefined ||
                        predicate === undefined ||
                        object === undefined
                    ) {
                        return null;
                    }
                    return [subject.toString(), predicate.toString(), object.toString()];
                }
                return null;
            })
            .filter(Boolean);
    }

    function normalizeClusterMap(rawClusters) {
        const map = new Map();
        if (!rawClusters) {
            return map;
        }
        if (rawClusters instanceof Map) {
            rawClusters.forEach((members, key) => {
                map.set(key.toString(), ensureArray(members).map(member => member.toString()));
            });
            return map;
        }
        const entries = Array.isArray(rawClusters)
            ? rawClusters
            : Object.entries(rawClusters);

        entries.forEach(entry => {
            if (!entry) {
                return;
            }
            if (Array.isArray(entry)) {
                const [key, members] = entry;
                if (key === undefined) {
                    return;
                }
                map.set(key.toString(), ensureArray(members).map(member => member.toString()));
            } else if (typeof entry === 'object' && entry.id !== undefined) {
                map.set(entry.id.toString(), ensureArray(entry.members).map(member => member.toString()));
            }
        });

        return map;
    }

    function normalizeClusterObject(rawClusters) {
        if (!rawClusters) {
            return {};
        }

        if (rawClusters instanceof Map) {
            const result = {};
            rawClusters.forEach((members, key) => {
                if (key === undefined || key === null) {
                    return;
                }
                result[key.toString()] = ensureArray(members).map(member => member.toString());
            });
            return result;
        }

        const entries = Array.isArray(rawClusters)
            ? rawClusters
            : Object.entries(rawClusters);

        const result = {};
        entries.forEach(entry => {
            if (!entry) {
                return;
            }
            if (Array.isArray(entry)) {
                const [key, members] = entry;
                if (key === undefined || key === null) {
                    return;
                }
                result[key.toString()] = ensureArray(members).map(member => member.toString());
            } else if (typeof entry === 'object' && entry.id !== undefined && entry.id !== null) {
                result[entry.id.toString()] = ensureArray(entry.members).map(member => member.toString());
            } else if (typeof entry === 'object') {
                Object.entries(entry).forEach(([key, members]) => {
                    if (key === undefined || key === null) {
                        return;
                    }
                    result[key.toString()] = ensureArray(members).map(member => member.toString());
                });
            }
        });

        return result;
    }

    function setStatus(message, type = 'info') {
        // no operation per now
    }

    function sanitizeGraphForBackend(rawGraph) {
        if (!rawGraph || typeof rawGraph !== 'object') {
            return rawGraph;
        }

        if (rawGraph.graph && typeof rawGraph.graph === 'object') {
            return {
                ...rawGraph,
                graph: sanitizeGraphForBackend(rawGraph.graph),
            };
        }

        const entityClustersSource = rawGraph.entity_clusters ?? rawGraph.entityClusters ?? rawGraph.entityclusters ?? {};
        const edgeClustersSource = rawGraph.edge_clusters ?? rawGraph.edgeClusters ?? rawGraph.edgeclusters ?? {};

        const normalizedEntityClusters = normalizeClusterObject(entityClustersSource);
        const normalizedEdgeClusters = normalizeClusterObject(edgeClustersSource);

        const payload = {
            ...rawGraph,
            entity_clusters: normalizedEntityClusters,
            edge_clusters: normalizedEdgeClusters,
        };

        if ('entityClusters' in payload) {
            payload.entityClusters = normalizedEntityClusters;
        }
        if ('entityclusters' in payload) {
            payload.entityclusters = normalizedEntityClusters;
        }
        if ('edgeClusters' in payload) {
            payload.edgeClusters = normalizedEdgeClusters;
        }
        if ('edgeclusters' in payload) {
            payload.edgeclusters = normalizedEdgeClusters;
        }

        return payload;
    }

    function normalizeGraphPayload(raw) {
        if (!raw || typeof raw !== 'object') {
            throw new Error('Graph JSON must be an object.');
        }
        if (raw.graph && typeof raw.graph === 'object') {
            return normalizeGraphPayload(raw.graph);
        }
        if (Array.isArray(raw.nodes) && Array.isArray(raw.edges) && raw.stats) {
            return raw;
        }

        const entities = ensureArray(raw.entities ?? raw.Nodes ?? raw.nodes ?? [])
            .map(entity => entity?.toString?.() ?? '')
            .filter(Boolean);

        const relationSource =
            raw.relations ??
            raw.Relations ??
            raw.relationships ??
            raw.triples ??
            raw.Triples ??
            raw.edges ??
            [];

        let relations = normalizeRelations(relationSource);
        if (
            !relations.length &&
            raw.relations &&
            typeof raw.relations === 'object' &&
            'data' in raw.relations
        ) {
            relations = normalizeRelations(raw.relations.data);
        }

        const entityClusters = normalizeClusterMap(
            raw.entity_clusters ?? raw.entityClusters ?? raw.entityclusters ?? []
        );

        const edgeClusters = normalizeClusterMap(
            raw.edge_clusters ?? raw.edgeClusters ?? raw.edgeclusters ?? []
        );

        return {
            entities,
            relations,
            entityClusters,
            edgeClusters,
        };
    }

    function hslToHex(h, s, l) {
        const hue = (h % 1 + 1) % 1;
        const saturation = Math.max(0, Math.min(1, s));
        const lightness = Math.max(0, Math.min(1, l));

        const a = saturation * Math.min(lightness, 1 - lightness);
        const f = n => {
            const k = (n + hue * 12) % 12;
            const color = lightness - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
            return Math.round(255 * color)
                .toString(16)
                .padStart(2, '0');
        };

        return `#${f(0)}${f(8)}${f(4)}`;
    }

    function stringToColor(label) {
        const text = (label ?? '').toString();
        let hash = 0;
        for (let index = 0; index < text.length; index += 1) {
            hash = Math.imul(31, hash) + text.charCodeAt(index);
            hash >>>= 0; // ensure unsigned 32-bit
        }
        const byte1 = hash & 0xff;
        const byte2 = (hash >>> 8) & 0xff;
        const byte3 = (hash >>> 16) & 0xff;
        const hue = byte1 / 255;
        const saturation = 0.55 + (byte2 / 255) * 0.3;
        const lightness = 0.45 + (byte3 / 255) * 0.25;
        return hslToHex(hue, saturation, lightness);
    }

    function buildViewModelLocally(rawGraph) {
        const graph = normalizeGraphPayload(rawGraph) ?? {};
        if (Array.isArray(graph.nodes) && Array.isArray(graph.edges) && graph.stats) {
            return graph;
        }

        const entitiesInput = ensureArray(graph.entities);
        const relationsInput = Array.isArray(graph.relations)
            ? graph.relations
            : ensureArray(graph.relations);

        const entityClustersMap = graph.entityClusters instanceof Map
            ? graph.entityClusters
            : normalizeClusterMap(graph.entityClusters);

        const edgeClustersMap = graph.edgeClusters instanceof Map
            ? graph.edgeClusters
            : normalizeClusterMap(graph.edgeClusters);

        const allEntities = new Set(entitiesInput);
        relationsInput.forEach(([subject, , object]) => {
            if (subject) {
                allEntities.add(subject);
            }
            if (object) {
                allEntities.add(object);
            }
        });

        const entities = sortedIgnoreCase(allEntities);

        const clusterView = [];
        const entityMemberToCluster = new Map();
        const nodeColorLookup = new Map();

        entityClustersMap.forEach((members, representative) => {
            const memberSet = new Set(ensureArray(members));
            memberSet.add(representative);
            const orderedMembers = sortedIgnoreCase(memberSet);
            const color = stringToColor(`entity::${representative}`);
            clusterView.push({
                id: representative,
                label: representative,
                members: orderedMembers,
                size: orderedMembers.length,
                color,
            });
            orderedMembers.forEach(member => {
                entityMemberToCluster.set(member, representative);
                nodeColorLookup.set(member, color);
            });
        });

        if (!clusterView.length) {
            entities.forEach(entity => {
                nodeColorLookup.set(entity, stringToColor(`entity::${entity}`));
            });
        } else {
            entities.forEach(entity => {
                if (!nodeColorLookup.has(entity)) {
                    nodeColorLookup.set(entity, stringToColor(`entity::${entity}`));
                }
            });
        }

        const edgeClusterView = [];
        const edgeMemberToCluster = new Map();
        const edgeColorLookup = new Map();

        edgeClustersMap.forEach((members, representative) => {
            const memberSet = new Set(ensureArray(members));
            memberSet.add(representative);
            const orderedMembers = sortedIgnoreCase(memberSet);
            const color = stringToColor(`edge::${representative}`);
            edgeClusterView.push({
                id: representative,
                label: representative,
                members: orderedMembers,
                size: orderedMembers.length,
                color,
            });
            orderedMembers.forEach(member => {
                edgeMemberToCluster.set(member, representative);
                edgeColorLookup.set(member, color);
            });
        });

        const degree = new Map();
        const indegree = new Map();
        const outdegree = new Map();
        const predicateCounts = new Map();
        const adjacency = new Map();
        const nodeNeighbors = new Map();
        const nodeEdges = new Map();

        const ensureNodeRecord = node => {
            if (!degree.has(node)) {
                degree.set(node, 0);
            }
            if (!indegree.has(node)) {
                indegree.set(node, 0);
            }
            if (!outdegree.has(node)) {
                outdegree.set(node, 0);
            }
            if (!adjacency.has(node)) {
                adjacency.set(node, new Set());
            }
            if (!nodeNeighbors.has(node)) {
                nodeNeighbors.set(node, new Set());
            }
            if (!nodeEdges.has(node)) {
                nodeEdges.set(node, { incoming: [], outgoing: [] });
            }
        };

        entities.forEach(entity => {
            ensureNodeRecord(entity);
        });

        const edgesView = [];
        relationsInput.forEach(([subject, predicate, object], index) => {
            const source = subject ?? '';
            const target = object ?? '';
            const edgePredicate = predicate ?? '';

            ensureNodeRecord(source);
            ensureNodeRecord(target);

            predicateCounts.set(
                edgePredicate,
                (predicateCounts.get(edgePredicate) || 0) + 1
            );

            degree.set(source, (degree.get(source) || 0) + 1);
            degree.set(target, (degree.get(target) || 0) + 1);
            outdegree.set(source, (outdegree.get(source) || 0) + 1);
            indegree.set(target, (indegree.get(target) || 0) + 1);

            adjacency.get(source).add(target);
            adjacency.get(target).add(source);
            nodeNeighbors.get(source).add(target);
            nodeNeighbors.get(target).add(source);

            const edgeId = `e${index}`;
            const cluster = edgeMemberToCluster.get(edgePredicate) || null;
            const color =
                edgeColorLookup.get(edgePredicate) ||
                stringToColor(`predicate::${edgePredicate}`);
            if (!edgeColorLookup.has(edgePredicate)) {
                edgeColorLookup.set(edgePredicate, color);
            }

            edgesView.push({
                id: edgeId,
                source,
                target,
                predicate: edgePredicate,
                cluster,
                color,
                tooltip: `${source} \u2014${edgePredicate}\u2192 ${target}`,
            });

            nodeEdges.get(source).outgoing.push(edgeId);
            nodeEdges.get(target).incoming.push(edgeId);
        });

        const isolatedEntities = entities.filter(
            entity => (degree.get(entity) || 0) === 0
        );

        const components = (() => {
            const visited = new Set();
            const result = [];
            entities.forEach(entity => {
                if (visited.has(entity)) {
                    return;
                }
                const queue = [entity];
                visited.add(entity);
                const members = [];
                while (queue.length) {
                    const current = queue.shift();
                    members.push(current);
                    (adjacency.get(current) || new Set()).forEach(neighbor => {
                        if (!visited.has(neighbor)) {
                            visited.add(neighbor);
                            queue.push(neighbor);
                        }
                    });
                }
                result.push({
                    size: members.length,
                    members: sortedIgnoreCase(members),
                });
            });
            result.sort((a, b) => {
                if (b.size !== a.size) {
                    return b.size - a.size;
                }
                const firstA = a.members[0] ?? '';
                const firstB = b.members[0] ?? '';
                return compareIgnoreCase(firstA, firstB);
            });
            return result;
        })();

        const nodesView = entities.map(entity => {
            const cluster = entityMemberToCluster.get(entity) || null;
            const nodeDegree = degree.get(entity) || 0;
            return {
                id: entity,
                label: entity,
                cluster,
                color: nodeColorLookup.get(entity) || '#64748b',
                degree: nodeDegree,
                indegree: indegree.get(entity) || 0,
                outdegree: outdegree.get(entity) || 0,
                isRepresentative: cluster ? cluster === entity : false,
                radius: 18 + Math.min(nodeDegree, 8) * 2,
                neighbors: sortedIgnoreCase(nodeNeighbors.get(entity) || []),
                edgeIds: nodeEdges.get(entity) || { incoming: [], outgoing: [] },
            };
        });

        const topEntities = nodesView
            .map(node => ({
                label: node.label,
                degree: node.degree,
                indegree: node.indegree,
                outdegree: node.outdegree,
                cluster: node.cluster,
            }))
            .sort((a, b) => {
                if (b.degree !== a.degree) {
                    return b.degree - a.degree;
                }
                return compareIgnoreCase(a.label, b.label);
            })
            .slice(0, 10);

        const topRelations = Array.from(predicateCounts.entries())
            .map(([predicate, count]) => ({
                predicate,
                count,
                cluster: edgeMemberToCluster.get(predicate) || null,
                color:
                    edgeColorLookup.get(predicate) ||
                    stringToColor(`predicate::${predicate}`),
            }))
            .sort((a, b) => {
                if (b.count !== a.count) {
                    return b.count - a.count;
                }
                return compareIgnoreCase(a.predicate, b.predicate);
            })
            .slice(0, 10);

        const totalDegree = entities.reduce(
            (sum, entity) => sum + (degree.get(entity) || 0),
            0
        );

        const stats = {
            entities: entities.length,
            relations: edgesView.length,
            relationTypes: predicateCounts.size,
            entityClusters: clusterView.length,
            edgeClusters: edgeClusterView.length,
            isolatedEntities: isolatedEntities.length,
            components: components.length,
            averageDegree: entities.length
                ? Math.round((totalDegree / entities.length) * 100) / 100
                : 0,
            density:
                entities.length > 1
                    ? Math.round(
                        (edgesView.length /
                            (entities.length * (entities.length - 1)) +
                            Number.EPSILON) *
                        1000
                    ) / 1000
                    : 0,
        };

        const relationRecords = edgesView.map((edge, index) => {
            const relation = relationsInput[index] || [];
            return {
                source: relation[0] ?? edge.source,
                predicate: relation[1] ?? edge.predicate,
                target: relation[2] ?? edge.target,
                edgeId: edge.id,
                color: edge.color,
            };
        });

        return {
            nodes: nodesView,
            edges: edgesView,
            clusters: clusterView,
            edgeClusters: edgeClusterView,
            topEntities,
            topRelations,
            stats,
            isolatedEntities,
            components,
            relations: relationRecords,
        };
    }

    const viewer = document.getElementById('viewer');
    const placeholder = document.getElementById('placeholder');
    const viewerWrapper = document.querySelector('.viewer-wrapper');
    const floatingActions = document.getElementById('floatingActions');
    const downloadButton = document.getElementById('downloadGraph');
    const refreshButton = document.getElementById('refreshView');

    const graphFileInput = document.getElementById('graphFile');
    const graphDropZone = document.getElementById('graphDropZone');
    const exampleSelect = document.getElementById('exampleGraph');
    const exampleLink = document.getElementById('exampleLink');
    const exampleStatus = document.getElementById('exampleStatus');

    const apiKeyInput = document.getElementById('apiKey');
    const modelSelect = document.getElementById('model');
    const chunkSizeInput = document.getElementById('chunkSize');
    const temperatureInput = document.getElementById('temperature');
    const clusterToggle = document.getElementById('clusterToggle');
    const contextInput = document.getElementById('context');
    const sourceText = document.getElementById('sourceText');
    const textFileInput = document.getElementById('textFile');

    const generateButton = document.getElementById('generateButton');
    const clearTextButton = document.getElementById('clearTextButton');

    console.log('[kg-gen] Element refs:', {
        apiKeyInput: !!apiKeyInput,
        modelSelect: !!modelSelect,
        chunkSizeInput: !!chunkSizeInput,
        temperatureInput: !!temperatureInput,
        clusterToggle: !!clusterToggle,
        contextInput: !!contextInput
    });

    const refreshCallbacks = [];
    const exampleMetadata = new Map();

    // localStorage keys for caching form inputs
    const CACHE_KEYS = {
        apiKey: 'kg-gen-api-key',
        model: 'kg-gen-model',
        chunkSize: 'kg-gen-chunk-size',
        temperature: 'kg-gen-temperature',
        cluster: 'kg-gen-cluster',
        context: 'kg-gen-context'
    };

    // Load cached form values from localStorage
    function loadCachedInputs() {
        console.log('[kg-gen] Loading cached inputs...');
        try {
            // Load API key
            const cachedApiKey = localStorage.getItem(CACHE_KEYS.apiKey);
            console.log('[kg-gen] Cached API key exists:', !!cachedApiKey);
            if (cachedApiKey && apiKeyInput) {
                apiKeyInput.value = cachedApiKey;
                console.log('[kg-gen] API key loaded');
            }

            // Load model
            const cachedModel = localStorage.getItem(CACHE_KEYS.model);
            console.log('[kg-gen] Cached model:', cachedModel);
            if (cachedModel && modelSelect) {
                modelSelect.value = cachedModel;
                console.log('[kg-gen] Model loaded:', modelSelect.value);
            }

            // Load chunk size
            const cachedChunkSize = localStorage.getItem(CACHE_KEYS.chunkSize);
            if (cachedChunkSize && chunkSizeInput) {
                chunkSizeInput.value = cachedChunkSize;
            }

            // Load temperature
            const cachedTemperature = localStorage.getItem(CACHE_KEYS.temperature);
            if (cachedTemperature && temperatureInput) {
                temperatureInput.value = cachedTemperature;
            }

            // Load cluster toggle
            const cachedCluster = localStorage.getItem(CACHE_KEYS.cluster);
            if (cachedCluster !== null && clusterToggle) {
                clusterToggle.checked = cachedCluster === 'true';
            }

            // Load context
            const cachedContext = localStorage.getItem(CACHE_KEYS.context);
            if (cachedContext && contextInput) {
                contextInput.value = cachedContext;
            }

            console.log('[kg-gen] Cached inputs loaded successfully');
        } catch (error) {
            console.warn('[kg-gen] Failed to load cached inputs:', error);
        }
    }

    // Save form input to localStorage
    function saveCachedInput(key, value) {
        try {
            if (value !== undefined && value !== null) {
                localStorage.setItem(key, String(value));
            } else {
                localStorage.removeItem(key);
            }
        } catch (error) {
            console.warn('[kg-gen] Failed to save cached input:', key, error);
        }
    }

    // Setup input change listeners for caching
    function setupInputCaching() {
        // API key caching
        if (apiKeyInput) {
            apiKeyInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.apiKey, apiKeyInput.value);
            });
        }

        // Model caching
        if (modelSelect) {
            modelSelect.addEventListener('change', () => {
                saveCachedInput(CACHE_KEYS.model, modelSelect.value);
            });
        }

        // Chunk size caching
        if (chunkSizeInput) {
            chunkSizeInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.chunkSize, chunkSizeInput.value);
            });
        }

        // Temperature caching
        if (temperatureInput) {
            temperatureInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.temperature, temperatureInput.value);
            });
        }

        // Cluster toggle caching
        if (clusterToggle) {
            clusterToggle.addEventListener('change', () => {
                saveCachedInput(CACHE_KEYS.cluster, clusterToggle.checked);
            });
        }

        // Context caching
        if (contextInput) {
            contextInput.addEventListener('input', () => {
                saveCachedInput(CACHE_KEYS.context, contextInput.value);
            });
        }

        // Enhanced mutual exclusion with visual feedback
        const sourceTextHint = document.getElementById('sourceTextHint');
        const textFileHint = document.getElementById('textFileHint');

        window.updateInputStates = function updateInputStates() {
            const hasText = sourceText && sourceText.value.trim();
            const hasFile = textFileInput && textFileInput.files && textFileInput.files.length > 0;
            const hasContent = hasText || hasFile;

            // Update input states
            if (sourceText) {
                sourceText.disabled = hasFile;
            }
            if (textFileInput) {
                textFileInput.disabled = hasText;
            }

            // Update hint visibility
            if (sourceTextHint) {
                if (hasFile) {
                    sourceTextHint.classList.add('show');
                } else {
                    sourceTextHint.classList.remove('show');
                }
            }
            if (textFileHint) {
                if (hasText) {
                    textFileHint.classList.add('show');
                } else {
                    textFileHint.classList.remove('show');
                }
            }

            // Update clear button style
            if (clearTextButton) {
                if (hasContent) {
                    clearTextButton.classList.add('active');
                } else {
                    clearTextButton.classList.remove('active');
                }
            }
        }

        if (textFileInput) {
            textFileInput.addEventListener('change', updateInputStates);
        }

        if (sourceText) {
            sourceText.addEventListener('input', updateInputStates);
        }

        // Initial state
        updateInputStates();
    }

    const modelDefaultTemperature = new Map([
        ['openai/gpt-5', 1.0],
        ['openai/gpt-5-nano', 1.0],
        ['openai/gpt-5-mini', 1.0],
        ['openai/gpt-5-large', 1.0],
        ['openai/gpt-4.1', 0.0],
        ['openai/gpt-4.1-mini', 0.0],
        ['openai/gpt-4.1-nano', 0.0],
        ['openai/gpt-4o', 0.0],
        ['openai/gpt-4o-mini', 0.0],
        ['openai/gpt-4o-mini-omni-math', 0.0],
    ]);

    function inferDefaultTemperature(modelId) {
        if (!modelId) {
            return 0;
        }
        if (modelDefaultTemperature.has(modelId)) {
            return modelDefaultTemperature.get(modelId);
        }
        if (modelId.includes('gpt-5')) {
            return 1.0;
        }
        return 0.0;
    }

    function applyDefaultTemperature(modelId, userTriggered = false) {
        if (!temperatureInput) {
            return;
        }
        const currentValue = temperatureInput.value.trim();
        if (!userTriggered && currentValue) {
            return;
        }
        const temperature = inferDefaultTemperature(modelId);
        temperatureInput.value = temperature.toString();
    }

    let templateHtml = null;
    let activeUrl = null;
    let lastGraphPayload = null;
    let lastViewModel = null;
    let isGenerating = false;
    let hasLoadedGraph = false;

    function resetViewer() {
        if (activeUrl) {
            URL.revokeObjectURL(activeUrl);
            activeUrl = null;
        }
        viewer.setAttribute('hidden', 'hidden');
        viewer.removeAttribute('src');
        placeholder.removeAttribute('hidden');
        placeholder.style.display = 'flex';
        floatingActions.setAttribute('hidden', 'hidden');
        refreshCallbacks.length = 0;
        hasLoadedGraph = false;
        if (viewerWrapper) {
            viewerWrapper.classList.remove('graph-loaded');
        }
    }

    // Global loading system - always shows full-screen overlay
    function showGlobalLoading(title, message) {
        showLoadingInPlaceholder(title, message);
    }

    function hideGlobalLoading() {
        hideLoadingInPlaceholder();
    }

    // Legacy functions for backward compatibility - now use global loading
    function showLoadingInViewer(title, message) {
        showGlobalLoading(title, message);
    }

    function hideLoadingInViewer() {
        hideGlobalLoading();
    }

    function showLoadingInPlaceholder(title, message) {
        // Remove any existing loading overlay first
        const existingOverlay = document.getElementById('kg-gen-loading-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }

        // Create loading overlay that covers entire screen
        const loadingOverlay = document.createElement('div');
        loadingOverlay.id = 'kg-gen-loading-overlay';

        // Set all styles directly on the element to ensure they're applied
        Object.assign(loadingOverlay.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            right: '0',
            bottom: '0',
            width: '100vw',
            height: '100vh',
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '1000001',
            padding: '1rem',
            pointerEvents: 'auto',
            overflow: 'hidden',
            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        });

        loadingOverlay.innerHTML = `
            <div class="loading-card" style="
                background: white;
                border-radius: 8px;
                padding: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                max-width: 300px;
                width: 100%;
                margin: auto;
                pointer-events: none;
            ">
                <div class="loading-spinner" style="
                    width: 32px;
                    height: 32px;
                    border: 3px solid #e5e7eb;
                    border-top: 3px solid #3b82f6;
                    border-radius: 50%;
                    animation: kg-spinner-spin 1s linear infinite;
                    margin: 0 auto 1rem;
                "></div>
                <h3 style="
                    margin: 0 0 0.5rem;
                    font-size: 1.125rem;
                    font-weight: 600;
                    color: #111827;
                    word-break: break-word;
                ">${title}</h3>
                <p style="
                    margin: 0;
                    color: #6b7280;
                    font-size: 0.875rem;
                    word-break: break-word;
                ">${message}</p>
            </div>
            <style>
                @keyframes kg-spinner-spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        `;

        // Add event listeners to prevent any clicks from going through
        loadingOverlay.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }, true);

        loadingOverlay.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }, true);

        loadingOverlay.addEventListener('keydown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }, true);

        // Insert at the very end of the body
        document.body.appendChild(loadingOverlay);

        // Hide the placeholder to avoid showing duplicate loading text
        if (placeholder) {
            placeholder.setAttribute('hidden', 'hidden');
            placeholder.style.display = 'none';
        }
    }

    function hideLoadingInPlaceholder() {
        // Remove the full-screen loading overlay
        const loadingOverlay = document.getElementById('kg-gen-loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.remove();
        }

        placeholder.innerHTML = '';
        if (!hasLoadedGraph) {
            placeholder.setAttribute('hidden', 'hidden');
            placeholder.style.display = 'flex';
        }
    }

    function confirmGraphReplacement(action) {
        if (!hasLoadedGraph || !isGenerating) {
            return true;
        }
        return confirm(
            'A graph is currently loaded. Proceeding will replace it.\n\n' +
            'Continue with ' + action + '?'
        );
    }

    function prepareDownload(graphJson) {
        if (!graphJson) {
            downloadButton.setAttribute('hidden', 'hidden');
            refreshButton.setAttribute('hidden', 'hidden');
            floatingActions.setAttribute('hidden', 'hidden');
            return;
        }
        lastGraphPayload = graphJson;
        downloadButton.removeAttribute('hidden');
        refreshButton.removeAttribute('hidden');
        floatingActions.removeAttribute('hidden');
    }

    function updateExampleLink(meta) {
        if (meta && meta.wiki_url) {
            exampleLink.href = meta.wiki_url;
            const label = meta.title ? `View ${meta.title} on Wikipedia` : 'View on Wikipedia';
            exampleLink.textContent = label;
            exampleLink.removeAttribute('hidden');
        } else {
            exampleLink.setAttribute('hidden', 'hidden');
            exampleLink.href = '#';
            exampleLink.textContent = 'View on Wikipedia';
        }
    }

    async function loadExamples() {
        if (!exampleSelect) {
            return;
        }

        exampleSelect.disabled = true;
        exampleMetadata.clear();
        exampleSelect.innerHTML = '';
        const loadingOption = document.createElement('option');
        loadingOption.value = '';
        loadingOption.textContent = 'Loading samples...';
        exampleSelect.appendChild(loadingOption);
        exampleStatus.textContent = 'Loading sample graphs...';
        updateExampleLink(null);

        try {
            const response = await fetch('/api/examples');
            const items = await response.json();
            if (!response.ok) {
                throw new Error(Array.isArray(items?.detail) ? items.detail.join('; ') : items?.detail || 'Failed to load sample list');
            }

            if (!Array.isArray(items) || !items.length) {
                const emptyOption = document.createElement('option');
                emptyOption.value = '';
                emptyOption.textContent = 'No samples available';
                exampleSelect.innerHTML = '';
                exampleSelect.appendChild(emptyOption);
                exampleStatus.textContent = 'Sample graphs are not available right now.';
                return;
            }

            exampleSelect.innerHTML = '';
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'Pick a sample graph...';
            exampleSelect.appendChild(placeholderOption);

            items.forEach(item => {
                if (!item || !item.slug) {
                    return;
                }
                exampleMetadata.set(item.slug, item);
                const option = document.createElement('option');
                option.value = item.slug;
                option.textContent = item.title || item.slug;
                exampleSelect.appendChild(option);
            });

            if (exampleMetadata.size === 0) {
                const fallbackOption = document.createElement('option');
                fallbackOption.value = '';
                fallbackOption.textContent = 'Samples unavailable';
                exampleSelect.innerHTML = '';
                exampleSelect.appendChild(fallbackOption);
                exampleStatus.textContent = 'Sample graphs are not available right now.';
                return;
            }

            exampleSelect.disabled = false;
            exampleStatus.textContent = 'Select an example to load it instantly.';

            // Auto-select and load the first example
            if (items.length > 0) {
                const firstExample = items[0];
                exampleSelect.value = firstExample.slug;

                // Automatically load the first example
                const meta = exampleMetadata.get(firstExample.slug);
                updateExampleLink(meta);
                const title = meta?.title || firstExample.slug;
                exampleStatus.textContent = `Loading ${title}...`;
                setStatus(`Loading example graph: ${title}...`);

                showLoadingInViewer('Loading Example', `Loading ${title}...`);
                if (!hasLoadedGraph) {
                    resetViewer();
                }

                exampleSelect.disabled = true;
                fetch(`/api/examples/${firstExample.slug}`)
                    .then(async response => {
                        let payload;
                        try {
                            payload = await response.json();
                        } catch (parseError) {
                            if (response.ok) {
                                throw new Error('Example payload is not valid JSON');
                            }
                            throw new Error(`Request failed (${response.status})`);
                        }

                        if (!response.ok) {
                            const message = payload?.detail || payload?.error || `Failed to load example (${response.status})`;
                            throw new Error(message);
                        }

                        return handleGraphData(payload);
                    })
                    .then(() => {
                        exampleStatus.textContent = `Loaded ${title}.`;
                    })
                    .catch(error => {
                        console.error('[kg-gen] Failed to load example graph', error);
                        setStatus(`Failed to load example '${title}': ${error.message}`, 'error');
                        exampleStatus.textContent = 'Could not load the selected sample.';
                        hideLoadingInViewer();
                    })
                    .finally(() => {
                        exampleSelect.disabled = exampleMetadata.size === 0;
                    });
            }
        } catch (error) {
            console.error('[kg-gen] Failed to load example graphs', error);
            exampleSelect.innerHTML = '';
            const fallbackOption = document.createElement('option');
            fallbackOption.value = '';
            fallbackOption.textContent = 'Samples unavailable';
            exampleSelect.appendChild(fallbackOption);
            exampleStatus.textContent = 'Could not load sample graphs.';
        }
    }

    function extractSummary(viewModel) {
        if (!viewModel || !viewModel.stats) {
            return 'Graph rendered.';
        }
        const stats = viewModel.stats;
        const summary = [];
        if (typeof stats.entities === 'number') summary.push(`${stats.entities} entities`);
        if (typeof stats.relations === 'number') summary.push(`${stats.relations} relations`);
        if (typeof stats.relationTypes === 'number') summary.push(`${stats.relationTypes} relation types`);
        return `Graph ready: ${summary.join(', ')}.`;
    }

    function sanitizeJson(jsonValue) {
        return JSON.stringify(jsonValue, null, 2).replace(/<\/script>/gi, '<\\/script>');
    }

    async function loadTemplate() {
        if (templateHtml) {
            return templateHtml;
        }
        setStatus('Loading visualization template...');
        try {
            const response = await fetch('/template');
            if (!response.ok) {
                throw new Error('Template fetch failed');
            }
            templateHtml = await response.text();
            setStatus('Template ready. Drop or generate a graph.');
        } catch (error) {
            setStatus(`Failed to load template: ${error.message}`, 'error');
            throw error;
        }
        return templateHtml;
    }

    async function renderView(viewModel, graphForDownload) {
        console.info('[kg-gen] Rendering view model');
        showLoadingInViewer('Rendering Graph', 'Building visualization...');
        await loadTemplate();
        const html = templateHtml.replace('<!--DATA-->', `\n${sanitizeJson(viewModel)}\n`);
        if (activeUrl) {
            URL.revokeObjectURL(activeUrl);
        }
        activeUrl = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
        viewer.src = activeUrl;
        viewer.removeAttribute('hidden');
        placeholder.setAttribute('hidden', 'hidden');
        placeholder.style.display = 'none';
        if (viewerWrapper) {
            viewerWrapper.classList.add('graph-loaded');
        }
        prepareDownload(graphForDownload || viewModel);
        lastViewModel = viewModel;
        setStatus(extractSummary(viewModel), 'success');
        refreshCallbacks.length = 0;
        refreshCallbacks.push(() => renderView(lastViewModel, lastGraphPayload));
        hasLoadedGraph = true;

        // Wait for iframe to be ready, then show initializing with global loading
        viewer.onload = () => {
            // Update global loading to show "Initializing Graph"
            showLoadingInViewer('Initializing Graph', 'Setting up the knowledge graph visualization.');

            // Keep global loading for graph initialization
            setTimeout(() => {
                hideLoadingInViewer();
            }, 1500); // Give time for graph initialization
        };

        // Notify sidebar manager about the new graph data
        if (window.sidebarManager) {
            setTimeout(() => {
                window.sidebarManager.handleGraphDataReady(viewModel);
            }, 500); // Small delay to ensure iframe is ready
        }
    }

    function readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = event => resolve(event.target?.result ?? '');
            reader.onerror = () => reject(new Error('File read failed'));
            reader.readAsText(file);
        });
    }

    async function handleGraphData(rawJson) {
        if (!rawJson) {
            throw new Error('Empty JSON payload');
        }

        const isViewModel = rawJson.nodes && rawJson.edges && rawJson.stats;
        if (isViewModel) {
            console.info('[kg-gen] Detected precomputed view model');
            setStatus('Rendering supplied visualization...');
            await renderView(rawJson, rawJson);
            return;
        }

        setStatus('Preparing graph data...');
        console.info('[kg-gen] Sending graph data to /api/graph/view');

        let remoteError = null;
        const backendPayload = sanitizeGraphForBackend(rawJson);
        try {
            const response = await fetch('/api/graph/view', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(backendPayload)
            });

            let payload;
            try {
                payload = await response.json();
            } catch (parseError) {
                if (!response.ok) {
                    throw new Error(`Backend responded with ${response.status}`);
                }
                throw parseError;
            }

            if (!response.ok) {
                console.warn('[kg-gen] Graph preparation failed', payload);
                let message = payload && typeof payload === 'object'
                    ? payload.detail || payload.error || payload.message
                    : null;
                if (Array.isArray(payload?.detail)) {
                    message = payload.detail
                        .map(item => item.msg || item.message || JSON.stringify(item))
                        .join('; ');
                }
                throw new Error(message || 'Failed to prepare graph');
            }

            console.info('[kg-gen] Graph preparation succeeded');
            await renderView(payload.view, payload.graph);
            return;
        } catch (error) {
            remoteError = error;
            console.warn('[kg-gen] Falling back to local rendering', error);
        }

        try {
            const viewModel = buildViewModelLocally(rawJson);
            console.info('[kg-gen] Graph rendered locally');
            await renderView(viewModel, rawJson);
            setStatus(`${extractSummary(viewModel)} (rendered locally)`, 'success');
        } catch (fallbackError) {
            console.error('[kg-gen] Local conversion failed', fallbackError);
            const combinedMessage = fallbackError?.message || remoteError?.message || 'Unknown error';
            setStatus(`Could not load graph: ${combinedMessage}`, 'error');
            resetViewer();
            throw fallbackError;
        }
    }

    async function handleGraphFile(file) {
        if (!file) {
            return;
        }

        if (!confirmGraphReplacement('file upload')) {
            return;
        }

        setStatus(`Reading ${file.name}...`);
        console.info('[kg-gen] Reading uploaded graph file', file.name);

        // Hide mobile sidebar and show loading with proper timing
        if (window.sidebarManager && window.sidebarManager.isMobile) {
            window.sidebarManager.hideMobileSidebar();
            // Give sidebar animation time to complete before showing loading
            setTimeout(() => {
                showLoadingInViewer('Loading Graph', `Reading ${file.name}...`);
            }, 150);
        } else {
            showLoadingInViewer('Loading Graph', `Reading ${file.name}...`);
        }

        try {
            const contents = await readFile(file);
            const json = JSON.parse(contents);
            await handleGraphData(json);
        } catch (error) {
            console.error(error);
            setStatus(`Could not load graph: ${error.message}`, 'error');
            hideLoadingInViewer();
            resetViewer();
        }
    }

    async function generateGraph() {
        const apiKey = apiKeyInput.value.trim();
        const pastedText = sourceText.value.trim();
        const textFile = textFileInput.files?.[0];
        const chunkSizeValue = chunkSizeInput.value.trim();
        const temperatureValue = temperatureInput.value.trim();

        if (!apiKey) {
            setStatus('Enter your OpenAI API key to generate a graph.', 'error');
            return;
        }

        if (!pastedText && !textFile) {
            setStatus('Provide some text or upload a .txt file.', 'error');
            return;
        }

        if (!confirmGraphReplacement('graph generation')) {
            return;
        }

        isGenerating = true;
        generateButton.disabled = true;
        generateButton.textContent = 'Generating...';

        const formData = new FormData();
        formData.append('api_key', apiKey);
        formData.append('model', modelSelect.value);
        formData.append('cluster', clusterToggle.checked ? 'true' : 'false');
        if (pastedText) {
            formData.append('source_text', pastedText);
        }
        if (textFile) {
            formData.append('text_file', textFile);
        }
        if (contextInput.value.trim()) {
            formData.append('context', contextInput.value.trim());
        }
        if (chunkSizeValue) {
            formData.append('chunk_size', chunkSizeValue);
        }
        if (temperatureValue) {
            formData.append('temperature', temperatureValue);
        }

        setStatus('Generating graph with KGGen...');
        console.info('[kg-gen] Submitting generate request', {
            model: modelSelect.value,
            cluster: clusterToggle.checked,
            hasText: Boolean(pastedText),
            hasFile: Boolean(textFile)
        });

        showLoadingInViewer('Generating Graph', 'Running KGGen on your text. This may take a few minutes...');
        if (!hasLoadedGraph) {
            resetViewer();
        }

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                body: formData
            });

            const rawBody = await response.text();
            let payload = null;
            if (rawBody) {
                try {
                    payload = JSON.parse(rawBody);
                } catch (parseError) {
                    console.warn('[kg-gen] Response was not valid JSON', parseError);
                }
            }

            if (!response.ok) {
                console.warn('[kg-gen] Generation endpoint returned error', payload || rawBody);
                const message =
                    (payload && (payload.detail || payload.error || payload.message)) ||
                    rawBody?.trim() ||
                    `Generation failed (${response.status})`;
                throw new Error(message);
            }

            if (!payload || typeof payload !== 'object') {
                throw new Error('Generation succeeded but returned invalid response.');
            }

            console.info('[kg-gen] Generation succeeded');
            await renderView(payload.view, payload.graph);
        } catch (error) {
            console.error(error);
            setStatus(`Generation failed: ${error.message}`, 'error');
            hideLoadingInViewer();
        } finally {
            isGenerating = false;
            generateButton.disabled = false;
            generateButton.textContent = 'Generate graph';
        }
    }

    graphDropZone.addEventListener('click', event => {
        event.preventDefault();
        graphFileInput.click();
    });

    graphDropZone.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            graphFileInput.click();
        }
    });

    graphFileInput.addEventListener('change', event => {
        const [file] = event.target.files;
        handleGraphFile(file);
        graphFileInput.value = '';
    });

    if (exampleSelect) {
        exampleSelect.addEventListener('change', async event => {
            const slug = event.target.value;
            if (!slug) {
                exampleStatus.textContent = exampleMetadata.size
                    ? 'Select an example to load it instantly.'
                    : exampleStatus.textContent;
                updateExampleLink(null);
                return;
            }

            if (!confirmGraphReplacement('example loading')) {
                exampleSelect.value = '';
                return;
            }

            const meta = exampleMetadata.get(slug) || null;
            updateExampleLink(meta);
            const title = meta?.title || slug;
            exampleStatus.textContent = `Loading ${title}...`;
            setStatus(`Loading example graph: ${title}...`);

            // Hide mobile sidebar and show loading with proper timing
            if (window.sidebarManager && window.sidebarManager.isMobile) {
                window.sidebarManager.hideMobileSidebar();
                // Give sidebar animation time to complete before showing loading
                setTimeout(() => {
                    showLoadingInViewer('Loading Example', `Loading ${title}...`);
                }, 150);
            } else {
                showLoadingInViewer('Loading Example', `Loading ${title}...`);
            }
            if (!hasLoadedGraph) {
                resetViewer();
            }

            exampleSelect.disabled = true;
            try {
                const response = await fetch(`/api/examples/${slug}`);
                let payload;
                try {
                    payload = await response.json();
                } catch (parseError) {
                    if (response.ok) {
                        throw new Error('Example payload is not valid JSON');
                    }
                    throw new Error(`Request failed (${response.status})`);
                }

                if (!response.ok) {
                    const message = payload?.detail || payload?.error || `Failed to load example (${response.status})`;
                    throw new Error(message);
                }

                await handleGraphData(payload);
                exampleStatus.textContent = `Loaded ${title}.`;
            } catch (error) {
                console.error('[kg-gen] Failed to load example graph', error);
                setStatus(`Failed to load example '${title}': ${error.message}`, 'error');
                exampleStatus.textContent = 'Could not load the selected sample.';
                hideLoadingInViewer();
            } finally {
                exampleSelect.disabled = exampleMetadata.size === 0;
            }
        });
    }

    if (modelSelect) {
        applyDefaultTemperature(modelSelect.value, false);
        modelSelect.addEventListener('change', event => {
            applyDefaultTemperature(event.target.value, true);
        });
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        graphDropZone.addEventListener(eventName, event => {
            event.preventDefault();
            event.stopPropagation();
            graphDropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        graphDropZone.addEventListener(eventName, event => {
            event.preventDefault();
            event.stopPropagation();
            graphDropZone.classList.remove('dragover');
        });
    });

    graphDropZone.addEventListener('drop', event => {
        const files = event.dataTransfer?.files;
        if (files && files.length) {
            console.info('[kg-gen] File dropped', files[0].name);
            handleGraphFile(files[0]);
        }
    });

    generateButton.addEventListener('click', event => {
        event.preventDefault();
        generateGraph();
    });

    clearTextButton.addEventListener('click', event => {
        event.preventDefault();
        sourceText.value = '';
        textFileInput.value = '';
        // Update states after clearing
        if (window.updateInputStates) {
            window.updateInputStates();
        }
        // Note: Cached inputs (api key, model, chunk size, temperature, cluster, context) are preserved
        setStatus('Text inputs cleared. Cached settings preserved.');
    });

    downloadButton.addEventListener('click', () => {
        if (!lastGraphPayload) {
            return;
        }
        const blob = new Blob([JSON.stringify(lastGraphPayload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'graph.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    });

    refreshButton.addEventListener('click', () => {
        if (!refreshCallbacks.length) {
            return;
        }
        const latest = refreshCallbacks[refreshCallbacks.length - 1];
        latest?.();
    });

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            console.log('[kg-gen] DOM loaded, initializing caching...');
            loadCachedInputs();
            setupInputCaching();
        });
    } else {
        // DOM is already loaded
        console.log('[kg-gen] DOM already loaded, initializing caching...');
        loadCachedInputs();
        setupInputCaching();
    }

    loadTemplate().catch(() => {
        /* Status already updated in loadTemplate */
    });
    loadExamples().catch(() => {
        /* Errors handled inside loadExamples */
    });
})();
