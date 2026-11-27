/**
 * GeoMind WebSocket Client - JavaScript/Node.js Example
 *
 * This example demonstrates how to connect to the GeoMind WebSocket API
 * from a Node.js application or browser.
 *
 * Usage:
 *   node examples/ws_client.js
 *   node examples/ws_client.js --url ws://localhost:8000/ws/connect --token <JWT>
 *   node examples/ws_client.js --url ws://localhost:8000/ws/connect --api-key <API_KEY>
 *
 * In browser:
 *   <script src="examples/ws_client.js"></script>
 *   <script>
 *     const client = new GeomindWSClient({ url: 'ws://localhost:8000/ws/connect' });
 *     client.connect().then(() => client.subscribe(['detections', 'detections:confirmed']));
 *   </script>
 */

class GeomindWSClient {
  /**
   * @param {Object} options
   * @param {string} options.url - WebSocket URL (e.g., 'ws://localhost:8000/ws/connect')
   * @param {string} [options.token] - JWT access token
   * @param {string} [options.apiKey] - API key
   * @param {string[]} [options.channels] - Channels to subscribe to
   * @param {Object} [options.filters] - Subscription filters
   * @param {number} [options.heartbeatInterval=30000] - Heartbeat interval in ms
   */
  constructor(options = {}) {
    this.url = options.url || 'ws://localhost:8000/ws/connect';
    this.token = options.token;
    this.apiKey = options.apiKey;
    this.channels = options.channels || ['detections', 'detections:confirmed', 'cameras'];
    this.filters = options.filters || {};
    this.heartbeatInterval = options.heartbeatInterval || 30000;

    this.ws = null;
    this.running = false;
    this.messageId = 0;
    this.subscribedChannels = new Set();
    this.heartbeatTimer = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;

    // Event handlers
    this.onOpen = null;
    this.onClose = null;
    this.onError = null;
    this.onDetection = null;
    this.onDetectionConfirmed = null;
    this.onDetectionPending = null;
    this.onCameraStatus = null;
    this.onMessage = null;
  }

  /**
   * Build WebSocket URL with authentication
   * @returns {string}
   */
  _buildUrl() {
    const params = new URLSearchParams();
    if (this.token) {
      params.set('token', this.token);
    } else if (this.apiKey) {
      params.set('token', this.apiKey);
    }

    const query = params.toString();
    return query ? `${this.url}?${query}` : this.url;
  }

  /**
   * Generate next message ID
   * @returns {string}
   */
  _nextId() {
    this.messageId++;
    return `msg_${this.messageId}`;
  }

  /**
   * Send a message
   * @param {Object} message
   */
  _send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }

  /**
   * Connect to WebSocket server
   * @returns {Promise<void>}
   */
  async connect() {
    const wsUrl = this._buildUrl();
    console.log(`Connecting to ${wsUrl}`);

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('Connected successfully');
        this.reconnectAttempts = 0;
        this.running = true;
        this._startHeartbeat();
        this.subscribe(this.channels, this.filters);
        if (this.onOpen) this.onOpen();
        resolve();
      };

      this.ws.onclose = (event) => {
        console.log(`Disconnected: ${event.code} ${event.reason}`);
        this._stopHeartbeat();
        this.running = false;
        if (this.onClose) this.onClose(event);

        // Attempt reconnection
        if (this.running && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
          console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
          setTimeout(() => this.connect(), delay);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (this.onError) this.onError(error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this._handleMessage(message);
        } catch (e) {
          console.warn('Invalid JSON message:', event.data);
        }
      };
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    this.running = false;
    this._stopHeartbeat();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Start heartbeat timer
   */
  _startHeartbeat() {
    this._stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.heartbeat();
    }, this.heartbeatInterval);
  }

  /**
   * Stop heartbeat timer
   */
  _stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Subscribe to channels
   * @param {string[]} channels
   * @param {Object} [filters]
   */
  subscribe(channels, filters = {}) {
    const message = {
      type: 'subscribe',
      id: this._nextId(),
      timestamp: new Date().toISOString(),
      channels,
      filters,
    };
    this._send(message);
    console.log(`Subscribed to: ${channels.join(', ')}`);
  }

  /**
   * Unsubscribe from channels
   * @param {string[]} channels
   */
  unsubscribe(channels) {
    const message = {
      type: 'unsubscribe',
      id: this._nextId(),
      timestamp: new Date().toISOString(),
      channels,
    };
    this._send(message);
    console.log(`Unsubscribed from: ${channels.join(', ')}`);
  }

  /**
   * Send heartbeat/ping
   * @param {Object} [payload]
   */
  heartbeat(payload = {}) {
    const message = {
      type: 'heartbeat',
      id: this._nextId(),
      timestamp: new Date().toISOString(),
      payload,
    };
    this._send(message);
  }

  /**
   * Handle incoming message
   * @param {Object} message
   */
  _handleMessage(message) {
    const type = message.type;

    // Call generic message handler
    if (this.onMessage) this.onMessage(message);

    switch (type) {
      case 'detection':
        this._handleDetection(message);
        break;
      case 'detection_confirmed':
        this._handleDetectionConfirmed(message);
        break;
      case 'detection_pending':
        this._handleDetectionPending(message);
        break;
      case 'camera_status':
        this._handleCameraStatus(message);
        break;
      case 'error':
        this._handleError(message);
        break;
      case 'ack':
        this._handleAck(message);
        break;
      case 'subscribed':
        this._handleSubscribed(message);
        break;
      case 'unsubscribed':
        this._handleUnsubscribed(message);
        break;
      default:
        console.warn(`Unknown message type: ${type}`);
    }
  }

  /**
   * Handle detection event
   * @param {Object} message
   */
  _handleDetection(message) {
    console.log(
      `🔥 DETECTION: camera=${message.camera_id} ` +
      `class=${message.class_name} ` +
      `conf=${message.confidence.toFixed(2)} ` +
      `status=${message.status} ` +
      `bbox=${JSON.stringify(message.bbox)} ` +
      `frame=${message.frame_id}`
    );
    if (this.onDetection) this.onDetection(message);
  }

  /**
   * Handle confirmed detection event
   * @param {Object} message
   */
  _handleDetectionConfirmed(message) {
    console.log(
      `✅ CONFIRMED: camera=${message.camera_id} ` +
      `class=${message.class_name} ` +
      `max_conf=${message.max_confidence.toFixed(2)} ` +
      `avg_conf=${message.avg_confidence.toFixed(2)} ` +
      `frames=${message.frame_count} ` +
      `event_id=${message.event_id}`
    );
    if (this.onDetectionConfirmed) this.onDetectionConfirmed(message);
  }

  /**
   * Handle pending detection event
   * @param {Object} message
   */
  _handleDetectionPending(message) {
    console.log(
      `⏳ PENDING: camera=${message.camera_id} ` +
      `class=${message.class_name} ` +
      `conf=${message.confidence.toFixed(2)} ` +
      `frame=${message.frame_id} ` +
      `event_id=${message.event_id}`
    );
    if (this.onDetectionPending) this.onDetectionPending(message);
  }

  /**
   * Handle camera status update
   * @param {Object} message
   */
  _handleCameraStatus(message) {
    console.log(
      `📷 CAMERA: ${message.camera_id} ` +
      `status=${message.status} ` +
      `frames=${message.total_frames} ` +
      `errors=${message.total_errors} ` +
      `fps=${message.fps ?? 'N/A'}`
    );
    if (this.onCameraStatus) this.onCameraStatus(message);
  }

  /**
   * Handle error message
   * @param {Object} message
   */
  _handleError(message) {
    console.error(`❌ ERROR: ${message.code} - ${message.message}`);
  }

  /**
   * Handle acknowledgment
   * @param {Object} message
   */
  _handleAck(message) {
    if (!message.success) {
      console.warn(`⚠️ ACK failed: ${message.message || 'Unknown error'}`);
    } else {
      console.debug(`✓ ACK: ${message.message || 'OK'}`);
    }
  }

  /**
   * Handle subscription confirmation
   * @param {Object} message
   */
  _handleSubscribed(message) {
    (message.channels || []).forEach(ch => this.subscribedChannels.add(ch));
    console.log(`✓ Subscribed: ${(message.channels || []).join(', ')}`);
  }

  /**
   * Handle unsubscription confirmation
   * @param {Object} message
   */
  _handleUnsubscribed(message) {
    (message.channels || []).forEach(ch => this.subscribedChannels.delete(ch));
    console.log(`✓ Unsubscribed: ${(message.channels || []).join(', ')}`);
  }
}

// Export for both Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { GeomindWSClient };
} else if (typeof window !== 'undefined') {
  window.GeomindWSClient = GeomindWSClient;
}

// CLI entry point for Node.js
if (typeof require !== 'undefined' && require.main === module) {
  const args = process.argv.slice(2);
  const options = {
    url: 'ws://localhost:8000/ws/connect',
    channels: ['detections', 'detections:confirmed', 'cameras'],
    filters: {},
    heartbeatInterval: 30000,
  };

  // Simple argument parsing
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--url':
        options.url = args[++i];
        break;
      case '--token':
        options.token = args[++i];
        break;
      case '--api-key':
        options.apiKey = args[++i];
        break;
      case '--channels':
        options.channels = args[++i].split(',');
        break;
      case '--filters':
        options.filters = JSON.parse(args[++i]);
        break;
      case '--heartbeat':
        options.heartbeatInterval = parseInt(args[++i]) * 1000;
        break;
      case '--help':
      case '-h':
        console.log(`
GeoMind WebSocket Client - JavaScript/Node.js

Usage:
  node examples/ws_client.js [options]

Options:
  --url <url>           WebSocket URL (default: ws://localhost:8000/ws/connect)
  --token <token>       JWT access token
  --api-key <key>       API key
  --channels <list>     Comma-separated channels (default: detections,detections:confirmed,cameras)
  --filters <json>      JSON filters (e.g., '{"camera_id":"cam_1"}')
  --heartbeat <sec>     Heartbeat interval in seconds (default: 30)
  --help, -h            Show this help

Examples:
  node examples/ws_client.js
  node examples/ws_client.js --token <JWT>
  node examples/ws_client.js --api-key <KEY> --channels detections
  node examples/ws_client.js --filters '{"camera_id":"cam_1","min_confidence":0.5}'
        `);
        process.exit(0);
    }
  }

  // Validate auth
  if (options.token && options.apiKey) {
    console.error('Cannot use both --token and --api-key');
    process.exit(1);
  }

  const client = new GeomindWSClient(options);

  // Handle shutdown
  process.on('SIGINT', () => {
    console.log('\nShutting down...');
    client.disconnect();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    client.disconnect();
    process.exit(0);
  });

  client.connect().catch(err => {
    console.error('Failed to connect:', err);
    process.exit(1);
  });
}