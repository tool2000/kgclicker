class SidebarManager {
    constructor() {
        this.isCollapsed = false;
        this.currentMode = 'setup';
        this.lastGraphData = null;
        this.init();
    }

    init() {
        this.createToggleButton();
        this.setupEventListeners();
        this.setupIframeMessaging();
        this.setupModeSwitching();
    }

    createToggleButton() {
        const button = document.createElement('button');
        button.id = 'sidebarToggle';
        button.className = 'sidebar-toggle';
        button.innerHTML = '‹';
        button.title = 'Toggle sidebar';

        const layout = document.querySelector('.layout');
        layout.appendChild(button);
    }

    setupEventListeners() {
        const toggleBtn = document.getElementById('sidebarToggle');

        toggleBtn.addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Listen for escape key to show sidebar
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isCollapsed) {
                this.showSidebar();
            }
        });
    }

    setupModeSwitching() {
        const modeTabs = document.querySelectorAll('.sidebar-mode-tab');
        const setupContent = document.getElementById('setupModeContent');
        const analysisContent = document.getElementById('analysisModeContent');

        modeTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const mode = tab.getAttribute('data-mode');
                this.switchMode(mode);
            });
        });
    }

    switchMode(mode) {
        if (this.currentMode === mode) return;

        this.currentMode = mode;
        const modeTabs = document.querySelectorAll('.sidebar-mode-tab');
        const setupContent = document.getElementById('setupModeContent');
        const analysisContent = document.getElementById('analysisModeContent');

        // Update tab states
        modeTabs.forEach(tab => {
            tab.classList.toggle('active', tab.getAttribute('data-mode') === mode);
        });

        // Update content visibility
        if (mode === 'setup') {
            setupContent.style.display = 'flex';
            analysisContent.style.display = 'none';
        } else if (mode === 'analysis') {
            setupContent.style.display = 'none';
            analysisContent.style.display = 'flex';

            // Auto-populate analysis if we have graph data
            if (this.lastGraphData) {
                this.updateAnalysisContent(this.lastGraphData);
            }
        }
    }

    setupIframeMessaging() {
        // Listen for messages from the iframe
        window.addEventListener('message', (event) => {
            if (event.data.type === 'graphDataReady') {
                this.handleGraphDataReady(event.data.data);
            } else if (event.data.type === 'selectionChanged') {
                this.handleSelectionChanged(event.data.selection);
            } else if (event.data.type === 'searchChanged') {
                this.handleSearchChanged(event.data.searchTerm);
            }
        });

        // Send initial state to iframe when it loads
        const iframe = document.getElementById('viewer');
        iframe.addEventListener('load', () => {
            setTimeout(() => {
                this.sendIframeMessage('disableInternalSidebar');
            }, 100);
        });
    }

    handleGraphDataReady(graphData) {
        this.lastGraphData = graphData;

        // Auto-switch to analysis mode when graph is loaded
        this.switchMode('analysis');
        this.updateAnalysisContent(graphData);
    }

    handleSelectionChanged(selection) {
        if (this.currentMode === 'analysis') {
            this.updateSelectionDetails(selection);
        }
    }

    handleSearchChanged(searchTerm) {
        if (this.currentMode === 'analysis') {
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {
                searchInput.value = searchTerm;
            }
        }
    }

    toggleSidebar() {
        if (this.isCollapsed) {
            this.showSidebar();
        } else {
            this.hideSidebar();
        }
    }

    hideSidebar() {
        const main = document.querySelector('main');
        const layout = document.querySelector('.layout');
        const toggleBtn = document.getElementById('sidebarToggle');

        main.style.transform = 'translateX(-100%)';
        layout.style.gridTemplateColumns = '0px 1fr';
        toggleBtn.innerHTML = '›';
        toggleBtn.style.left = '12px';

        this.isCollapsed = true;
    }

    showSidebar() {
        const main = document.querySelector('main');
        const layout = document.querySelector('.layout');
        const toggleBtn = document.getElementById('sidebarToggle');

        main.style.transform = 'translateX(0)';
        layout.style.gridTemplateColumns = '420px minmax(0, 1fr)';
        toggleBtn.innerHTML = '‹';
        toggleBtn.style.left = '432px';

        this.isCollapsed = false;
    }

    // Remove old iframe sidebar toggle method since we're using unified sidebar

    sendIframeMessage(type, data = {}) {
        const iframe = document.getElementById('viewer');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({ type, ...data }, '*');
        }
    }

    updateAnalysisContent(graphData) {
        const analysisContent = document.getElementById('analysisContent');
        const analysisPlaceholder = document.querySelector('.analysis-placeholder');

        if (!graphData || !graphData.stats) {
            analysisContent.style.display = 'none';
            analysisPlaceholder.style.display = 'block';
            return;
        }

        analysisContent.style.display = 'flex';
        analysisPlaceholder.style.display = 'none';

        // Update stats
        this.updateStats(graphData.stats);

        // Update lists
        this.updateEntityClusters(graphData.clusters || []);
        this.updateTopEntities(graphData.topEntities || []);
        this.updateTopRelations(graphData.topRelations || []);

        // Setup interactions
        this.setupAnalysisInteractions();
    }

    updateStats(stats) {
        const statsGrid = document.getElementById('statsGrid');
        if (!statsGrid) return;

        const entries = [
            { label: 'Entities', value: stats.entities },
            { label: 'Relations', value: stats.relations },
            { label: 'Relation types', value: stats.relationTypes },
            { label: 'Entity clusters', value: stats.entityClusters },
            { label: 'Edge clusters', value: stats.edgeClusters },
            { label: 'Isolated entities', value: stats.isolatedEntities },
            { label: 'Components', value: stats.components },
            { label: 'Average degree', value: stats.averageDegree }
        ];

        statsGrid.innerHTML = entries
            .map(entry => `
                <div class="stat-card">
                    <div class="stat-value">${entry.value}</div>
                    <div>${entry.label}</div>
                </div>
            `)
            .join('');
    }

    updateEntityClusters(clusters) {
        const container = document.getElementById('clusterList');
        if (!container) return;

        if (!clusters.length) {
            container.innerHTML = '<div class="meta">No cluster information provided.</div>';
            return;
        }

        container.innerHTML = clusters
            .map(cluster => `
                <button class="list-item cluster-item" data-cluster="${cluster.id}" style="border-left-color: ${cluster.color};">
                    <div>
                        <strong>${cluster.label}</strong>
                        <div class="meta">${cluster.size} members</div>
                    </div>
                    <span class="legend-swatch" style="background: ${cluster.color};"></span>
                </button>
            `)
            .join('');
    }

    updateTopEntities(topEntities) {
        const container = document.getElementById('topEntities');
        if (!container) return;

        const entityItems = topEntities.map(item => {
            const clusterLabel = item.cluster ? `${item.cluster}` : 'Unclustered';
            return `
                <div class="list-item">
                    <div>
                        <strong>${item.label}</strong>
                        <div class="meta">Cluster: ${clusterLabel}</div>
                    </div>
                    <div>${item.degree}</div>
                </div>
            `;
        });

        container.innerHTML = entityItems.join('') || '<div class="meta">No entities</div>';
    }

    updateTopRelations(topRelations) {
        const container = document.getElementById('topRelations');
        if (!container) return;

        const relationItems = topRelations.map(item => `
            <div class="list-item relation-item" data-predicate="${item.predicate}">
                <div>
                    <strong>${item.predicate}</strong>
                    <div class="meta">${item.count} relations</div>
                </div>
                <div class="pill" style="background: ${item.color}1a; border: 1px solid ${item.color};">${item.count}</div>
            </div>
        `);

        container.innerHTML = relationItems.join('') || '<div class="meta">No relations</div>';
    }

    updateSelectionDetails(selection) {
        const panel = document.getElementById('selectionDetails');
        if (!panel) return;

        if (!selection) {
            panel.innerHTML = 'Click a node or relation in the network to inspect details.';
            return;
        }

        if (selection.type === 'node') {
            const cluster = selection.cluster ? `Cluster: ${selection.cluster}` : 'Unclustered';
            const neighbors = selection.neighbors && selection.neighbors.length ? selection.neighbors.join(', ') : 'None';
            panel.innerHTML = `
                <h3>${selection.label}</h3>
                <ul>
                    <li>${cluster}</li>
                    <li>Total degree: ${selection.degree || 0}</li>
                    <li>Outgoing: ${selection.outdegree || 0}</li>
                    <li>Incoming: ${selection.indegree || 0}</li>
                    <li>Neighbors: ${neighbors}</li>
                </ul>
            `;
        } else if (selection.type === 'edge') {
            const cluster = selection.cluster ? `Cluster: ${selection.cluster}` : 'Unclustered';
            panel.innerHTML = `
                <h3>${selection.source} → ${selection.target}</h3>
                <ul>
                    <li>Relation: ${selection.predicate}</li>
                    <li>${cluster}</li>
                </ul>
            `;
        }
    }

    setupAnalysisInteractions() {
        // Setup search input
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            const debounce = (func, wait) => {
                let timeout;
                return function executedFunction(...args) {
                    const later = () => {
                        clearTimeout(timeout);
                        func(...args);
                    };
                    clearTimeout(timeout);
                    timeout = setTimeout(later, wait);
                };
            };

            const handleSearch = debounce(() => {
                this.sendIframeMessage('search', { term: searchInput.value.trim() });
            }, 300);

            searchInput.addEventListener('input', handleSearch);
            searchInput.addEventListener('keydown', (event) => {
                if (event.key === 'Escape') {
                    searchInput.value = '';
                    this.sendIframeMessage('search', { term: '' });
                }
            });
        }

        // Setup cluster interactions
        document.querySelectorAll('.cluster-item').forEach(item => {
            item.addEventListener('click', () => {
                const clusterId = item.getAttribute('data-cluster');
                this.sendIframeMessage('focusCluster', { clusterId });

                // Toggle active state
                const isActive = item.classList.contains('active');
                document.querySelectorAll('.cluster-item').forEach(el => el.classList.remove('active'));
                if (!isActive) {
                    item.classList.add('active');
                }
            });
        });

        // Setup relation interactions
        document.querySelectorAll('.relation-item').forEach(item => {
            item.addEventListener('click', () => {
                const predicate = item.getAttribute('data-predicate');
                this.sendIframeMessage('focusPredicate', { predicate });

                // Toggle active state
                const isActive = item.classList.contains('active');
                document.querySelectorAll('.relation-item').forEach(el => el.classList.remove('active'));
                if (!isActive) {
                    item.classList.add('active');
                }
            });
        });
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sidebarManager = new SidebarManager();
});