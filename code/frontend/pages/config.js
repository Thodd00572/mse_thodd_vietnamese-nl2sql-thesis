import { useState, useEffect } from 'react'
import Layout from '../components/Layout'
import { Settings, CheckCircle, AlertCircle, RefreshCw, Cloud, Server } from 'lucide-react'
import api from '../utils/api'

export default function ConfigPage() {
  const [config, setConfig] = useState({
    pipeline1_url: 'https://abnormally-direct-rhino.ngrok-free.app',
    pipeline2_url: 'https://abnormally-direct-rhino.ngrok-free.app'
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
                label="Pipeline 1 (PhoBERT-SQL)"
                url={status.pipeline1_url}
              />
              <StatusIndicator
                isHealthy={status.pipeline2_healthy}
                label="Pipeline 2 (PhoBERT + SQLCoder)"
                url={status.pipeline2_url}
              />
              
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h3 className="font-medium text-blue-900 mb-2 flex items-center">
                  <Cloud className="w-4 h-4 mr-2" />
                  Architecture Overview
                </h3>
                <div className="text-sm text-blue-800 space-y-1">
                  <p>• <strong>Local Server:</strong> Handles UI, database queries, and lightweight logic</p>
                  <p>• <strong>Colab Pipeline 1:</strong> PhoBERT-SQL model for direct Vietnamese → SQL</p>
                  <p>• <strong>Colab Pipeline 2:</strong> Vietnamese → English → SQL via MarianMT + SQLCoder</p>
                  <p>• <strong>Fallback:</strong> Local rule-based processing when Colab APIs are unavailable</p>
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
              <label htmlFor="pipeline1_url" className="block text-sm font-medium text-gray-700 mb-2">
                Pipeline 1 URL (PhoBERT-SQL)
              </label>
              <input
                type="url"
                id="pipeline1_url"
                value={config.pipeline1_url}
                onChange={(e) => setConfig(prev => ({ ...prev, pipeline1_url: e.target.value }))}
                placeholder="https://abnormally-direct-rhino.ngrok-free.app"
                className="input-field"
                disabled={saving}
              />
              <p className="mt-1 text-xs text-gray-500">
                Pipeline 1 is running at: https://abnormally-direct-rhino.ngrok-free.app (port 8000)
              </p>
            </div>

            <div>
              <label htmlFor="pipeline2_url" className="block text-sm font-medium text-gray-700 mb-2">
                Pipeline 2 URL (PhoBERT + SQLCoder)
              </label>
              <input
                type="url"
                id="pipeline2_url"
                value={config.pipeline2_url}
                onChange={(e) => setConfig(prev => ({ ...prev, pipeline2_url: e.target.value }))}
                placeholder="https://def456-8001.ngrok-free.app"
                className="input-field"
                disabled={saving}
              />
              <p className="mt-1 text-xs text-gray-500">
                Run Pipeline 2 Colab notebook to get your ngrok URL (port 8001)
              </p>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h3 className="font-medium text-yellow-800 mb-2">Setup Instructions:</h3>
              <ol className="text-sm text-yellow-700 space-y-1 list-decimal list-inside">
                <li>✅ Pipeline 1 is running at: https://abnormally-direct-rhino.ngrok-free.app</li>
                <li>🔄 Run Pipeline 2 Colab notebook to get the second ngrok URL (port 8001)</li>
                <li>📝 Paste both URLs above and click "Save Configuration"</li>
                <li>🔗 The system will automatically use Colab APIs when available, with local fallback</li>
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

        {/* Performance Benefits */}
        <div className="card mt-8 bg-gradient-to-r from-green-50 to-blue-50">
          <h3 className="font-semibold text-gray-900 mb-4">Performance Benefits</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-green-700 mb-2">With Colab GPU:</h4>
              <ul className="text-sm text-green-600 space-y-1">
                <li>• Pipeline 1: ~2-3 seconds</li>
                <li>• Pipeline 2: ~3-5 seconds</li>
                <li>• GPU acceleration for models</li>
                <li>• Parallel processing capability</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-blue-700 mb-2">Local Fallback:</h4>
              <ul className="text-sm text-blue-600 space-y-1">
                <li>• Pipeline 1: Rule-based (instant)</li>
                <li>• Pipeline 2: ~15-20 seconds (CPU)</li>
                <li>• Always available offline</li>
                <li>• No external dependencies</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}
