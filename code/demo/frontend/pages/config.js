import { useState, useEffect } from 'react'
import Layout from '../components/Layout'
import { Settings, CheckCircle, AlertCircle, RefreshCw, Cloud, Server } from 'lucide-react'
import api from '../utils/api'

export default function ConfigPage() {
  const [config, setConfig] = useState({
    base_url: 'https://abnormally-direct-rhino.ngrok-free.app',
    pipeline1_url: 'https://abnormally-direct-rhino.ngrok-free.app/p1',
    pipeline2_url: 'https://abnormally-direct-rhino.ngrok-free.app/p2',
    pipeline3_url: 'https://abnormally-direct-rhino.ngrok-free.app/p3'
  })
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    fetchStatus()
  }, [])

  const fetchStatus = async () => {
    setLoading(true)
    try {
      // Try Colab endpoint first with proper CORS handling
      console.log('Fetching status from Colab...')
      const colabResponse = await fetch('https://abnormally-direct-rhino.ngrok-free.app/config/colab/status', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        mode: 'cors'
      })
      
      console.log('Colab response status:', colabResponse.status)
      
      if (colabResponse.ok) {
        const colabData = await colabResponse.json()
        console.log('Colab data:', colabData)
        setStatus(colabData.status)
        if (colabData.status.pipeline1_url) {
          setConfig(prev => ({ ...prev, pipeline1_url: colabData.status.pipeline1_url }))
        }
        if (colabData.status.pipeline2_url) {
          setConfig(prev => ({ ...prev, pipeline2_url: colabData.status.pipeline2_url }))
        }
        return
      }
      
      // Fallback to local backend
      console.log('Falling back to local backend...')
      const response = await api.get('/api/config/colab/status')
      setStatus(response.data.status)
      if (response.data.status.pipeline1_url) {
        setConfig(prev => ({ ...prev, pipeline1_url: response.data.status.pipeline1_url }))
      }
      if (response.data.status.pipeline2_url) {
        setConfig(prev => ({ ...prev, pipeline2_url: response.data.status.pipeline2_url }))
      }
    } catch (error) {
      console.error('Failed to fetch status:', error)
      console.error('Error details:', error.message)
      // Set default status when both endpoints fail
      setStatus({
        pipeline1_healthy: false,
        pipeline2_healthy: false,
        pipeline1_url: config.pipeline1_url,
        pipeline2_url: config.pipeline2_url,
        colab_status: "disconnected"
      })
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      const response = await api.post('/config/colab', config)
      setStatus(response.data.status)
      alert('Configuration saved successfully!')
    } catch (error) {
      console.error('Failed to save config:', error)
      alert('Failed to save configuration: ' + (error.response?.data?.detail || error.message))
    } finally {
      setSaving(false)
    }
  }

  const StatusIndicator = ({ isHealthy, label, url }) => (
    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
      <div className="flex items-center space-x-3">
        <div className={`w-3 h-3 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`}></div>
        <span className="font-medium text-gray-900">{label}</span>
      </div>
      <div className="text-right">
        <div className="flex items-center space-x-2">
          {isHealthy ? (
            <CheckCircle className="w-4 h-4 text-green-500" />
          ) : (
            <AlertCircle className="w-4 h-4 text-red-500" />
          )}
          <span className={`text-sm ${isHealthy ? 'text-green-700' : 'text-red-700'}`}>
            {isHealthy ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        {url && (
          <div className="text-xs text-gray-500 mt-1 max-w-xs truncate">
            {url}
          </div>
        )}
      </div>
    </div>
  )

  return (
    <Layout>
      <div className="px-4 sm:px-0">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center">
            <Settings className="w-8 h-8 mr-3" />
            Colab Configuration
          </h1>
          <p className="mt-2 text-gray-600">
            Configure Google Colab API endpoints for hybrid cloud inference
          </p>
        </div>

        {/* Status Section */}
        <div className="card mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Connection Status</h2>
            <button
              onClick={fetchStatus}
              disabled={loading}
              className="btn-secondary flex items-center"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>

          {status ? (
            <div className="space-y-4">
              <StatusIndicator
                isHealthy={status.pipeline1_healthy}
                label="P1: mT5 Zero-Shot (EM: 16%, EX: 33%)"
                url={status.pipeline1_url}
              />
              <StatusIndicator
                isHealthy={status.pipeline2_healthy}
                label="P2: SQLCoder Zero-Shot (EM: 18%, EX: 22%)"
                url={status.pipeline2_url}
              />
              <StatusIndicator
                isHealthy={status.pipeline3_healthy}
                label="P3: Vanna AI RAG (EM: 43%, EX: 76%)"
                url={status.pipeline3_url}
              />
              
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h3 className="font-medium text-blue-900 mb-2 flex items-center">
                  <Cloud className="w-4 h-4 mr-2" />
                  Architecture Overview
                </h3>
                <div className="text-sm text-blue-800 space-y-1">
                  <p>• <strong>P1 (mT5 Zero-Shot):</strong> Direct Vietnamese→SQL with mT5 multilingual model (304ms latency, 4.8GB GPU)</p>
                  <p>• <strong>P2 (SQLCoder Zero-Shot):</strong> Schema-aware generation with SQLCoder-7B (1,763ms latency, 13.8GB GPU)</p>
                  <p>• <strong>P3 (Vanna AI RAG):</strong> ChromaDB + OpenAI GPT-4o with 98 training examples (1,779ms latency, 3.3GB GPU)</p>
                  <p>• <strong>Shared Domain:</strong> All pipelines use abnormally-direct-rhino.ngrok-free.app with separate paths (/p1, /p2, /p3)</p>
                  <p>• <strong>API Endpoints:</strong> /p1/generate, /p2/generate, /p3/generate (POST with {`{"query": "vietnamese text"}`})</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <Server className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <p className="text-gray-500">Loading status...</p>
            </div>
          )}
        </div>

        {/* Configuration Form */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">API Configuration</h2>
          
          <form onSubmit={handleSave} className="space-y-6">
            <div>
              <label htmlFor="base_url" className="block text-sm font-medium text-gray-700 mb-2">
                Shared ngrok Base URL
              </label>
              <input
                type="url"
                id="base_url"
                value={config.base_url}
                onChange={(e) => setConfig(prev => ({ ...prev, base_url: e.target.value }))}
                placeholder="https://abnormally-direct-rhino.ngrok-free.app"
                className="input-field"
                disabled={saving}
              />
              <p className="mt-1 text-xs text-gray-500">
                All three pipelines share this ngrok domain with different paths
              </p>
            </div>

            <div>
              <label htmlFor="pipeline1_url" className="block text-sm font-medium text-gray-700 mb-2">
                P1: mT5 Zero-Shot Endpoint
              </label>
              <input
                type="url"
                id="pipeline1_url"
                value={config.pipeline1_url}
                onChange={(e) => setConfig(prev => ({ ...prev, pipeline1_url: e.target.value }))}
                placeholder="https://abnormally-direct-rhino.ngrok-free.app/p1"
                className="input-field"
                disabled={saving}
              />
              <p className="mt-1 text-xs text-gray-500">
                Fast (304ms), lightweight (4.8GB), perfect reliability (100%)
              </p>
            </div>

            <div>
              <label htmlFor="pipeline2_url" className="block text-sm font-medium text-gray-700 mb-2">
                P2: SQLCoder Zero-Shot Endpoint
              </label>
              <input
                type="url"
                id="pipeline2_url"
                value={config.pipeline2_url}
                onChange={(e) => setConfig(prev => ({ ...prev, pipeline2_url: e.target.value }))}
                placeholder="https://abnormally-direct-rhino.ngrok-free.app/p2"
                className="input-field"
                disabled={saving}
              />
              <p className="mt-1 text-xs text-gray-500">
                Best for simple queries (67% EX), 13.8GB GPU required
              </p>
            </div>

            <div>
              <label htmlFor="pipeline3_url" className="block text-sm font-medium text-gray-700 mb-2">
                P3: Vanna AI RAG Endpoint
              </label>
              <input
                type="url"
                id="pipeline3_url"
                value={config.pipeline3_url}
                onChange={(e) => setConfig(prev => ({ ...prev, pipeline3_url: e.target.value }))}
                placeholder="https://abnormally-direct-rhino.ngrok-free.app/p3"
                className="input-field"
                disabled={saving}
              />
              <p className="mt-1 text-xs text-gray-500">
                Best overall (76% EX), handles all complexity levels, only 3.3GB GPU
              </p>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h3 className="font-medium text-yellow-800 mb-2">Setup Instructions:</h3>
              <ol className="text-sm text-yellow-700 space-y-1 list-decimal list-inside">
                <li>✅ All pipelines share domain: https://abnormally-direct-rhino.ngrok-free.app</li>
                <li>🔄 Run desired Colab notebooks (P1, P2, or P3) on port 8000</li>
                <li>📝 API sections are separated - skip if you only need training/evaluation</li>
                <li>🔗 Each pipeline auto-installs FastAPI dependencies before starting</li>
                <li>⚡ Use P1 for speed, P2 for simple queries, P3 for best accuracy</li>
              </ol>
            </div>

            <button
              type="submit"
              disabled={saving}
              className="btn-primary w-full flex items-center justify-center"
            >
              {saving ? (
                <>
                  <div className="loading-spinner mr-2"></div>
                  Saving...
                </>
              ) : (
                <>
                  <Settings className="w-4 h-4 mr-2" />
                  Save Configuration
                </>
              )}
            </button>
          </form>
        </div>

        {/* Performance Comparison */}
        <div className="card mt-8 bg-gradient-to-r from-green-50 to-blue-50">
          <h3 className="font-semibold text-gray-900 mb-4">Performance Comparison (300 Queries)</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-medium text-green-700 mb-2">P1: mT5 Zero-Shot</h4>
              <ul className="text-sm text-green-600 space-y-1">
                <li>• Latency: 304ms (fastest)</li>
                <li>• GPU Memory: 4.8GB</li>
                <li>• EM: 16% | EX: 33%</li>
                <li>• Reliability: 100%</li>
                <li>• Best for: Speed critical apps</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-blue-700 mb-2">P2: SQLCoder Zero-Shot</h4>
              <ul className="text-sm text-blue-600 space-y-1">
                <li>• Latency: 1,763ms</li>
                <li>• GPU Memory: 13.8GB (heaviest)</li>
                <li>• EM: 18% | EX: 22%</li>
                <li>• Reliability: 100%</li>
                <li>• Best for: Simple queries only</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-purple-700 mb-2">P3: Vanna AI RAG</h4>
              <ul className="text-sm text-purple-600 space-y-1">
                <li>• Latency: 1,779ms</li>
                <li>• GPU Memory: 3.3GB (lightest)</li>
                <li>• EM: 43% | EX: 76% (best)</li>
                <li>• Reliability: 87%</li>
                <li>• Best for: Production (all complexity)</li>
              </ul>
            </div>
          </div>
          <div className="mt-4 p-3 bg-white rounded-lg">
            <p className="text-sm text-gray-700">
              <strong>Recommendation:</strong> P3 Vanna AI is the only pipeline that handles simple, medium, and complex queries effectively. 
              Use P1 if latency is critical and accuracy can be sacrificed. Avoid P2 for production unless queries are guaranteed simple.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  )
}
