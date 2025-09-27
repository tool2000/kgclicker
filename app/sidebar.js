class SidebarManager {
    constructor() {
        this.isCollapsed = false;
        this.init();
    }

    init() {
        this.createToggleButton();
        this.setupEventListeners();
        this.setupIframeMessaging();
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
        const toggleGraphDetailsBtn = document.getElementById('toggleGraphDetails');

        toggleBtn.addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Add event listener for graph details toggle button
        if (toggleGraphDetailsBtn) {
            toggleGraphDetailsBtn.addEventListener('click', () => {
                this.toggleGraphDetails();
            });
        }

        // Listen for escape key to show sidebar
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isCollapsed) {
                this.showSidebar();
            }
        });
    }

    setupIframeMessaging() {
        // Listen for messages from the iframe to toggle its internal sidebar
        window.addEventListener('message', (event) => {
            if (event.data.type === 'toggleInternalSidebar') {
                this.toggleIframeSidebar();
            }
        });

        // Send initial state to iframe when it loads
        const iframe = document.getElementById('viewer');
        iframe.addEventListener('load', () => {
            setTimeout(() => {
                this.sendIframeMessage('setSidebarState', { collapsed: false });
            }, 100);
        });
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

    toggleIframeSidebar() {
        this.sendIframeMessage('toggleSidebar');
    }

    sendIframeMessage(type, data = {}) {
        const iframe = document.getElementById('viewer');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({ type, ...data }, '*');
        }
    }

    // Method to toggle iframe sidebar independently
    toggleGraphDetails() {
        this.sendIframeMessage('toggleSidebar');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sidebarManager = new SidebarManager();
});