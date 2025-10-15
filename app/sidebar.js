class SidebarManager {
    constructor() {
        this.isCollapsed = false;
        this.currentMode = 'setup';
        this.lastGraphData = null;
        this.isMobile = false;
        this.init();
    }

    init() {
        this.createToggleButton();
        this.setupEventListeners();
        this.setupIframeMessaging();
        this.setupModeSwitching();
        this.setupMobileMenu();
        this.checkMobile();
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
            // On mobile, also close sidebar with escape
            if (e.key === 'Escape' && this.isMobile) {
                this.hideMobileSidebar();
            }
        });

        // Listen for window resize to check mobile state
        window.addEventListener('resize', () => {
            this.checkMobile();
        });

        // Header global search wiring
        const globalSearch = document.getElementById('globalSearch');
        const globalSearchClear = document.getElementById('globalSearchClear');
        if (globalSearch) {
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
                this.sendIframeMessage('search', { term: globalSearch.value.trim() });
            }, 300);

            globalSearch.addEventListener('input', handleSearch);
            globalSearch.addEventListener('keydown', (event) => {
                if (event.key === 'Escape') {
                    globalSearch.value = '';
                    this.sendIframeMessage('search', { term: '' });
                }
            });

            if (globalSearchClear) {
                const toggleClearVisibility = () => {
                    globalSearchClear.style.opacity = globalSearch.value ? '1' : '0';
                };
                globalSearch.addEventListener('input', toggleClearVisibility);
                toggleClearVisibility();
            }
        }

        if (globalSearchClear && globalSearch) {
            globalSearchClear.addEventListener('click', () => {
                globalSearch.value = '';
                globalSearch.focus();
                this.sendIframeMessage('search', { term: '' });
            });
        }

    }

    setupMobileMenu() {
        const mobileMenuButton = document.getElementById('mobileMenuButton');
        if (mobileMenuButton) {
            mobileMenuButton.addEventListener('click', () => {
                this.toggleMobileSidebar();
            });
        }
    }

    checkMobile() {
        const wasMobile = this.isMobile;
        this.isMobile = window.innerWidth <= 768;
        const toggleBtn = document.getElementById('sidebarToggle');
        const placeholder = document.getElementById('placeholder');

        console.log('checkMobile called - placeholder found:', !!placeholder, 'isMobile:', this.isMobile);

        // Always update toggle button visibility based on mobile state
        if (toggleBtn) {
            if (this.isMobile) {
                toggleBtn.style.display = 'none';
            } else {
                toggleBtn.style.display = 'flex';
            }
        }

        // Update placeholder text and add mobile click handler
        if (placeholder) {
            if (this.isMobile) {
                placeholder.textContent = 'Tap here 👆 to pick your graph';
                document.querySelector('.viewer-wrapper').onclick = () => this.showMobileSidebar();
            } else {
                placeholder.textContent = 'Load or generate a graph to inspect it here.';
                document.querySelector('.viewer-wrapper').onclick = null;
            }
        } else {
            setTimeout(() => this.updatePlaceholderForMobile(), 100);
        }

        if (wasMobile !== this.isMobile) {
            // Reset sidebar state when switching between mobile/desktop
            const main = document.querySelector('main');
            const mobileMenuButton = document.getElementById('mobileMenuButton');

            if (this.isMobile) {
                // Entering mobile mode
                main.classList.remove('mobile-open');
                if (mobileMenuButton) {
                    mobileMenuButton.classList.remove('active');
                }
            } else {
                // Entering desktop mode
                main.classList.remove('mobile-open');
                if (mobileMenuButton) {
                    mobileMenuButton.classList.remove('active');
                }
                // Restore desktop sidebar state
                if (this.isCollapsed) {
                    this.hideSidebar();
                } else {
                    this.showSidebar();
                }
            }
        }
    }

    updatePlaceholderForMobile() {
        const placeholder = document.getElementById('placeholder');
        if (placeholder && this.isMobile) {
            placeholder.textContent = 'Tap here 👆 to pick your graph';
            document.querySelector('.viewer-wrapper').onclick = () => this.showMobileSidebar();
        }
    }

    toggleMobileSidebar() {
        const main = document.querySelector('main');
        const mobileMenuButton = document.getElementById('mobileMenuButton');

        if (main.classList.contains('mobile-open')) {
            this.hideMobileSidebar();
        } else {
            this.showMobileSidebar();
        }
    }

    showMobileSidebar() {
        const main = document.querySelector('main');
        const mobileMenuButton = document.getElementById('mobileMenuButton');

        main.classList.add('mobile-open');
        if (mobileMenuButton) {
            mobileMenuButton.classList.add('active');
        }
    }

    hideMobileSidebar() {
        const main = document.querySelector('main');
        const mobileMenuButton = document.getElementById('mobileMenuButton');

        main.classList.remove('mobile-open');
        if (mobileMenuButton) {
            mobileMenuButton.classList.remove('active');
        }
    }

    setupModeSwitching() {
        const modeTabs = document.querySelectorAll('.sidebar-mode-tab');
        const analysisContent = document.getElementById('analysisModeContent');
        const openContent = document.getElementById('openModeContent');
        const generateContent = document.getElementById('generateModeContent');

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
        const openContent = document.getElementById('openModeContent');
        const generateContent = document.getElementById('generateModeContent');
        const analysisContent = document.getElementById('analysisModeContent');

        // Update tab states
        modeTabs.forEach(tab => {
            tab.classList.toggle('active', tab.getAttribute('data-mode') === mode);
        });

        // Update content visibility
        if (mode === 'open') {
            openContent.style.display = 'flex';
            generateContent.style.display = 'none';
            analysisContent.style.display = 'none';
        } else if (mode === 'generate') {
            openContent.style.display = 'none';
            generateContent.style.display = 'flex';
            analysisContent.style.display = 'none';
        } else if (mode === 'analysis') {
            openContent.style.display = 'none';
            generateContent.style.display = 'none';
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

        // Close mobile sidebar when graph is picked on mobile
        if (this.isMobile) {
            this.hideMobileSidebar();
        }
    }

    handleSelectionChanged(selection) {
        if (this.currentMode === 'analysis') {
            this.updateSelectionDetails(selection);
        }
        
        // Update popover with selection data
        this.updatePopoverSelection(selection);
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
        // On mobile, use mobile sidebar toggle instead
        if (this.isMobile) {
            this.toggleMobileSidebar();
            return;
        }

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
            return `
                <div class="list-item entity-item" data-id="${item.label}">
                    <div>
                        <strong>${item.label}</strong>
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

    updatePopoverSelection(selection) {
        // Check if popover functions are available
        if (typeof window.updateSelectionPopover !== 'function') {
            return;
        }

        if (!selection) {
            window.updateSelectionPopover('Click a node or relation in the network to inspect details.');
            return;
        }

        // Create formatted content for the popover
        let popoverContent = '';
        
        if (selection.type === 'node') {
            const cluster = selection.cluster ? `Cluster: ${selection.cluster}` : 'Unclustered';
            const neighbors = selection.neighbors && selection.neighbors.length ? 
                selection.neighbors.slice(0, 3).join(', ') + (selection.neighbors.length > 3 ? '...' : '') : 'None';
            
            popoverContent = `
                <p class="selection-item-title"><strong>${selection.label}</strong></p>
                <div class="selection-item">
                    <p class="selection-item-characteristics"><strong>Type:</strong> Node</p>
                    <p class="selection-item-characteristics"><strong>Cluster:</strong> ${cluster}</p>
                    <p class="selection-item-characteristics"><strong>Degree:</strong> ${selection.degree || 0}</p>
                    <p class="selection-item-characteristics"><strong>Out:</strong> ${selection.outdegree || 0}</p>
                    <p class="selection-item-characteristics"><strong>In:</strong> ${selection.indegree || 0}</p>
                    <p class="selection-item-characteristics"><strong>Neighbors:</strong> ${neighbors}</p>
                </div>
            `;
        } else if (selection.type === 'edge') {
            const cluster = selection.cluster ? `Cluster: ${selection.cluster}` : 'Unclustered';
            
            popoverContent = `
                <p class="selection-item-title"><strong>${selection.source} → ${selection.target}</strong></p>
                <div class="selection-item">
                    <p class="selection-item-characteristics"><strong>Type:</strong> Relation</p>
                    <p class="selection-item-characteristics"><strong>Predicate:</strong> ${selection.predicate}</p>
                    <p class="selection-item-characteristics"><strong>Cluster:</strong> ${cluster}</p>
                </div>
            `;
        }

        window.updateSelectionPopover(popoverContent);
        
        // Automatically open the popover when selection changes
        // if (typeof window.openSelectionPopover === 'function') {
        //     window.openSelectionPopover();
        // }
    }

    setupAnalysisInteractions() {
        // Setup search input
        const searchInput = document.getElementById('globalSearch');
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
                console.log('handleSearch called with term:', searchInput.value.trim());
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

        // Setup top entities interactions
        document.querySelectorAll('.entity-item').forEach(item => {
            item.addEventListener('click', () => {
                const id = item.getAttribute('data-id');
                this.sendIframeMessage('focusEntity', { id });

                // Toggle active state
                const isActive = item.classList.contains('active');
                document.querySelectorAll('.entity-item').forEach(el => el.classList.remove('active'));
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