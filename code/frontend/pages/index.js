import { useState, useEffect } from 'react'
import Layout from '../components/Layout'
import { Search, Clock, Database, AlertCircle, CheckCircle, Download, Cloud, Settings } from 'lucide-react'
import Link from 'next/link'
import api from '../utils/api'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedPipeline, setSelectedPipeline] = useState('both')
  const [colabStatus, setColabStatus] = useState(null)
  const [processLogs, setProcessLogs] = useState([])
  const [showProcessView, setShowProcessView] = useState(false)

  useEffect(() => {
    fetchColabStatus()
  }, [])

  const fetchColabStatus = async () => {
    try {
      const response = await api.get('/config/colab/status')
      setColabStatus(response.data.status)
    } catch (error) {
      console.error('Failed to fetch Colab status:', error)
    }
  }

  const addProcessLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString()
    setProcessLogs(prev => [...prev, { timestamp, message, type }])
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setProcessLogs([])
    setShowProcessView(true)
    
    addProcessLog('🚀 Starting search process...', 'info')
    addProcessLog(`📝 Query: "${query.trim()}"`, 'info')
    addProcessLog(`🔧 Pipeline: ${selectedPipeline}`, 'info')
    
    try {
      addProcessLog('📡 Making API call to backend...', 'info')
      const response = await api.post('/api/search', {
        query: query.trim(),
        pipeline: selectedPipeline
      })
      
      addProcessLog('✅ API call successful', 'success')
      addProcessLog('📊 Processing results...', 'info')
      
      setResults(response.data)
      
      // Check for Colab connection errors
      const hasColabError = (response.data.pipeline1_result?.requires_colab) || 
                           (response.data.pipeline2_result?.requires_colab)
      
      if (hasColabError) {
        addProcessLog('⚠️ Colab server connection required!', 'error')
      }
      
      // Log pipeline results with detailed metrics
      if (response.data.pipeline1_result) {
        const p1 = response.data.pipeline1_result
        addProcessLog(`🔵 Pipeline 1: ${p1.success ? 'SUCCESS' : 'FAILED'}`, p1.success ? 'success' : 'error')
        addProcessLog(`⏱️ Pipeline 1 Execution Time: ${(p1.execution_time * 1000).toFixed(2)}ms`, 'info')
        if (p1.sql_query) {
          addProcessLog(`📝 Pipeline 1 SQL: ${p1.sql_query}`, 'info')
        }
        if (p1.metrics && p1.metrics.detailed_metrics) {
          const metrics = p1.metrics.detailed_metrics
          if (metrics.tokenization_time) {
            addProcessLog(`🔤 Tokenization: ${(metrics.tokenization_time * 1000).toFixed(1)}ms`, 'info')
          }
          if (metrics.phobert_inference_time) {
            addProcessLog(`🧠 PhoBERT Inference: ${(metrics.phobert_inference_time * 1000).toFixed(1)}ms`, 'info')
          }
          if (metrics.sql_generation_time) {
            addProcessLog(`⚙️ SQL Generation: ${(metrics.sql_generation_time * 1000).toFixed(1)}ms`, 'info')
          }
        }
        if (p1.error) {
          addProcessLog(`❌ Pipeline 1 Error: ${p1.error}`, 'error')
        }
        if (p1.requires_colab) {
          addProcessLog('🔗 Pipeline 1 requires Colab connection', 'error')
        }
      }
      
      if (response.data.pipeline2_result) {
        const p2 = response.data.pipeline2_result
        addProcessLog(`🟢 Pipeline 2: ${p2.success ? 'SUCCESS' : 'FAILED'}`, p2.success ? 'success' : 'error')
        addProcessLog(`⏱️ Pipeline 2 Total Time: ${(p2.execution_time * 1000).toFixed(2)}ms`, 'info')
        if (p2.english_query) {
          addProcessLog(`🌐 English Translation: ${p2.english_query}`, 'info')
        }
        if (p2.translation_time) {
          addProcessLog(`🔄 Translation Time: ${(p2.translation_time * 1000).toFixed(1)}ms`, 'info')
        }
        if (p2.sql_generation_time) {
          addProcessLog(`⚙️ SQL Generation Time: ${(p2.sql_generation_time * 1000).toFixed(1)}ms`, 'info')
        }
        if (p2.sql_query) {
          addProcessLog(`📝 Pipeline 2 SQL: ${p2.sql_query}`, 'info')
        }
        if (p2.error) {
          addProcessLog(`❌ Pipeline 2 Error: ${p2.error}`, 'error')
        }
        if (p2.requires_colab) {
          addProcessLog('🔗 Pipeline 2 requires Colab connection', 'error')
        }
      }
      
      if (!hasColabError) {
        addProcessLog('🎉 Search completed successfully!', 'success')
      }
      
    } catch (error) {
      console.error('Search error:', error)
      addProcessLog(`❌ API Error: ${error.message}`, 'error')
      
      if (error.response?.data?.detail) {
        addProcessLog(`📋 Error Details: ${error.response.data.detail}`, 'error')
      }
      
      if (error.response?.status) {
        addProcessLog(`🔢 HTTP Status: ${error.response.status}`, 'error')
      }
      
      setResults({
        error: error.response?.data?.detail || 'Search failed'
      })
    } finally {
      setLoading(false)
    }
  }

  const formatPrice = (price) => {
    return new Intl.NumberFormat('vi-VN', {
      style: 'currency',
      currency: 'VND'
    }).format(price)
  }

  const PipelineResult = ({ result, pipelineNumber }) => {
    if (!result) return null

    const bgColor = pipelineNumber === 1 ? 'bg-blue-50 border-blue-200' : 'bg-green-50 border-green-200'
    const accentColor = pipelineNumber === 1 ? 'text-blue-700' : 'text-green-700'

    return (
      <div className={`${bgColor} border rounded-xl p-6 space-y-4`}>
        <div className="flex items-center justify-between">
          <h3 className={`font-semibold ${accentColor} text-lg`}>
            {result.pipeline_name}
          </h3>
          <div className="flex items-center space-x-2">
            {result.success ? (
              <span className="status-success">
                <CheckCircle className="w-3 h-3 inline mr-1" />
                Success
              </span>
            ) : (
              <span className="status-error">
                <AlertCircle className="w-3 h-3 inline mr-1" />
                Error
              </span>
            )}
            <span className="text-sm text-gray-600 flex items-center">
              <Clock className="w-3 h-3 mr-1" />
              {(result.execution_time * 1000).toFixed(2)}ms
            </span>
          </div>
        </div>

        {result.english_query && (
          <div>
            <p className="text-sm font-medium text-gray-700">English Translation:</p>
            <p className="text-sm text-gray-600 bg-white p-2 rounded border">
              {result.english_query}
            </p>
          </div>
        )}

        <div>
          <p className="text-sm font-medium text-gray-700">Generated SQL:</p>
          <pre className="code-block text-xs mt-2">
            {result.sql_query || 'No SQL generated'}
          </pre>
        </div>

        {/* Detailed Metrics Display */}
        {result.metrics && (
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-sm font-medium text-gray-700 mb-2">Performance Metrics:</p>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-gray-600">Method:</span>
                <span className="ml-1 font-mono">{result.metrics.method || 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-600">Version:</span>
                <span className="ml-1 font-mono">{result.metrics.version || 'N/A'}</span>
              </div>
              {result.metrics.detailed_metrics && (
                <>
                  {result.metrics.detailed_metrics.tokenization_time && (
                    <div>
                      <span className="text-gray-600">Tokenization:</span>
                      <span className="ml-1 font-mono text-blue-600">
                        {(result.metrics.detailed_metrics.tokenization_time * 1000).toFixed(1)}ms
                      </span>
                    </div>
                  )}
                  {result.metrics.detailed_metrics.phobert_inference_time && (
                    <div>
                      <span className="text-gray-600">PhoBERT:</span>
                      <span className="ml-1 font-mono text-green-600">
                        {(result.metrics.detailed_metrics.phobert_inference_time * 1000).toFixed(1)}ms
                      </span>
                    </div>
                  )}
                  {result.metrics.detailed_metrics.sql_generation_time && (
                    <div>
                      <span className="text-gray-600">SQL Gen:</span>
                      <span className="ml-1 font-mono text-purple-600">
                        {(result.metrics.detailed_metrics.sql_generation_time * 1000).toFixed(1)}ms
                      </span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Pipeline 2 Specific Metrics */}
        {pipelineNumber === 2 && (result.translation_time || result.sql_generation_time) && (
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-sm font-medium text-gray-700 mb-2">Pipeline 2 Breakdown:</p>
            <div className="grid grid-cols-2 gap-3 text-xs">
              {result.translation_time && (
                <div>
                  <span className="text-gray-600">Translation:</span>
                  <span className="ml-1 font-mono text-blue-600">
                    {(result.translation_time * 1000).toFixed(1)}ms
                  </span>
                </div>
              )}
              {result.sql_generation_time && (
                <div>
                  <span className="text-gray-600">SQL Generation:</span>
                  <span className="ml-1 font-mono text-green-600">
                    {(result.sql_generation_time * 1000).toFixed(1)}ms
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {result.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <p className="text-sm text-red-700">
              <AlertCircle className="w-4 h-4 inline mr-1" />
              {result.error}
            </p>
          </div>
        )}

        {result.results && result.results.length > 0 && (
          <div>
            <p className="text-sm font-medium text-gray-700 mb-3">
              Results ({result.results.length} products):
            </p>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {result.results.map((product, idx) => (
                <div key={idx} className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 text-sm">
                        {product.name}
                      </h4>
                      <p className="text-xs text-gray-600 mt-1">
                        {product.description}
                      </p>
                      <div className="flex items-center space-x-4 mt-2">
                        <span className="text-sm font-semibold text-primary-600">
                          {formatPrice(product.price)}
                        </span>
                        <span className="text-xs text-gray-500">
                          {product.brand}
                        </span>
                        {product.rating && (
                          <span className="text-xs text-yellow-600">
                            ⭐ {product.rating} ({product.review_count} reviews)
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.results && result.results.length === 0 && !result.error && (
          <div className="text-center py-8 text-gray-500">
            <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No products found</p>
          </div>
        )}
      </div>
    )
  }

  return (
    <Layout>
      <div className="px-4 sm:px-0">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Vietnamese Product Search
              </h1>
              <p className="mt-2 text-gray-600">
                Compare two translation pipelines for Vietnamese to SQL conversion
              </p>
            </div>
            
            {/* Colab Status Widget */}
            <div className="flex items-center space-x-4">
              {colabStatus && (
                <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm">
                  <div className="flex items-center space-x-2 mb-2">
                    <Cloud className="w-4 h-4 text-gray-600" />
                    <span className="text-sm font-medium text-gray-700">Colab Status</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-1">
                      <div className={`w-2 h-2 rounded-full ${colabStatus.pipeline1_healthy ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span className="text-xs text-gray-600">P1</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className={`w-2 h-2 rounded-full ${colabStatus.pipeline2_healthy ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span className="text-xs text-gray-600">P2</span>
                    </div>
                  </div>
                </div>
              )}
              
              <Link href="/config" className="btn-secondary flex items-center">
                <Settings className="w-4 h-4 mr-2" />
                Configure
              </Link>
            </div>
          </div>
          
          {/* Colab Connection Error Banner */}
          {colabStatus && (!colabStatus.pipeline1_healthy || !colabStatus.pipeline2_healthy) && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-red-800">
                    ⚠️ Colab Server Connection Required
                  </p>
                  <p className="text-xs text-red-700 mt-1">
                    {!colabStatus.pipeline1_healthy && "Pipeline 1 requires Colab connection. "}
                    {!colabStatus.pipeline2_healthy && "Pipeline 2 requires Colab connection. "}
                    Search will fail without proper Colab server setup.
                  </p>
                </div>
                <Link href="/config" className="btn-primary text-xs px-3 py-1">
                  Connect to Colab
                </Link>
              </div>
            </div>
          )}
          
          {/* Additional error banner for when results show Colab requirement */}
          {results && (results.pipeline1_result?.requires_colab || results.pipeline2_result?.requires_colab) && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-red-800">
                    🚫 Search Failed - Colab Server Not Connected
                  </p>
                  <p className="text-xs text-red-700 mt-1">
                    Both pipelines require an active Colab server connection. Please configure and connect to your Colab server to enable Vietnamese NL2SQL processing.
                  </p>
                </div>
                <Link href="/config" className="btn-primary text-xs px-3 py-1">
                  Setup Colab Now
                </Link>
              </div>
            </div>
          )}
        </div>

        {/* Search Form */}
        <div className="card mb-8">
          <form onSubmit={handleSearch} className="space-y-4">
            <div>
              <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                Vietnamese Query
              </label>
              <input
                type="text"
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ví dụ: tìm iPhone giá rẻ, laptop Apple, tai nghe Samsung..."
                className="input-field"
                disabled={loading}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Pipeline Selection
              </label>
              <div className="flex space-x-4">
                {[
                  { value: 'both', label: 'Both Pipelines' },
                  { value: 'pipeline1', label: 'Pipeline 1 Only (PhoBERT-SQL)' },
                  { value: 'pipeline2', label: 'Pipeline 2 Only (PhoBERT + SQLCoder)' }
                ].map((option) => (
                  <label key={option.value} className="flex items-center">
                    <input
                      type="radio"
                      name="pipeline"
                      value={option.value}
                      checked={selectedPipeline === option.value}
                      onChange={(e) => setSelectedPipeline(e.target.value)}
                      className="mr-2"
                      disabled={loading}
                    />
                    <span className="text-sm text-gray-700">{option.label}</span>
                  </label>
                ))}
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="btn-primary w-full flex items-center justify-center"
            >
              {loading ? (
                <>
                  <div className="loading-spinner mr-2"></div>
                  Processing...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  Search Products
                </>
              )}
            </button>
          </form>
        </div>

        {/* Process View */}
        {showProcessView && (
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Process Log</h3>
              <button
                onClick={() => setShowProcessView(false)}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Hide
              </button>
            </div>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-64 overflow-y-auto">
              {processLogs.map((log, index) => (
                <div key={index} className={`mb-1 ${
                  log.type === 'error' ? 'text-red-400' : 
                  log.type === 'success' ? 'text-green-400' : 
                  'text-gray-300'
                }`}>
                  <span className="text-gray-500">[{log.timestamp}]</span> {log.message}
                </div>
              ))}
              {loading && (
                <div className="text-yellow-400 mb-1">
                  <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> ⏳ Processing...
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results */}
        {results && !results.error && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-900">
                Search Results
              </h2>
              <div className="text-sm text-gray-500">
                Query: "{results.vietnamese_query}"
              </div>
            </div>

            {/* Pipeline Results Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {results.pipeline1_result && (
                <PipelineResult result={results.pipeline1_result} pipelineNumber={1} />
              )}
              {results.pipeline2_result && (
                <PipelineResult result={results.pipeline2_result} pipelineNumber={2} />
              )}
            </div>

            {/* Comparison Summary */}
            {results.pipeline1_result && results.pipeline2_result && (
              <div className="card bg-gradient-to-r from-blue-50 to-green-50">
                <h3 className="font-semibold text-gray-900 mb-4">Pipeline Comparison</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Faster Pipeline</p>
                    <p className="font-semibold text-lg">
                      {results.pipeline1_result.execution_time < results.pipeline2_result.execution_time 
                        ? 'Pipeline 1' : 'Pipeline 2'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {Math.abs(results.pipeline1_result.execution_time - results.pipeline2_result.execution_time * 1000).toFixed(2)}ms difference
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Result Count</p>
                    <p className="font-semibold text-lg">
                      P1: {results.pipeline1_result.results.length} | P2: {results.pipeline2_result.results.length}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Success Status</p>
                    <div className="flex justify-center space-x-2">
                      <span className={results.pipeline1_result.success ? 'status-success' : 'status-error'}>
                        P1
                      </span>
                      <span className={results.pipeline2_result.success ? 'status-success' : 'status-error'}>
                        P2
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {results && results.error && (
          <div className="card bg-red-50 border-red-200">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
              <h3 className="font-semibold text-red-700">Search Error</h3>
            </div>
            <p className="mt-2 text-red-600">{results.error}</p>
          </div>
        )}

        {/* Sample Queries by Complexity */}
        <div className="card mt-8">
          <h3 className="font-semibold text-gray-900 mb-6">Sample Vietnamese Queries</h3>
          
          {/* Simple Queries */}
          <div className="mb-8">
            <div className="flex items-center mb-4">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
              <h4 className="font-medium text-green-700 text-lg">Simple Queries</h4>
              <span className="ml-2 text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">Basic keyword matching</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {[
                "Tìm tất cả balo nữ",
                "Hiển thị các giày thể thao nam có giá dưới 500k",
                "Tìm túi xách của thương hiệu Samsonite",
                "Cho tôi xem tất cả dép tổ ong",
                "Tìm giày boots nữ có rating trên 4 sao",
                "Hiển thị kính mát có giá từ 100k đến 300k",
                "Tìm balo laptop",
                "Cho tôi xem vali vải",
                "Tìm giày sandals nam"
              ].map((sampleQuery, idx) => (
                <button
                  key={`simple-${idx}`}
                  onClick={() => setQuery(sampleQuery)}
                  className="text-left p-3 bg-green-50 hover:bg-green-100 border border-green-200 rounded-lg transition-colors text-sm"
                  disabled={loading}
                >
                  "{sampleQuery}"
                </button>
              ))}
            </div>
          </div>

          {/* Medium Queries */}
          <div className="mb-8">
            <div className="flex items-center mb-4">
              <div className="w-3 h-3 bg-yellow-500 rounded-full mr-3"></div>
              <h4 className="font-medium text-yellow-700 text-lg">Medium Queries</h4>
              <span className="ml-2 text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full">Multiple conditions & OR logic</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                "Tìm giày thể thao thương hiệu Nike có giá dưới 1 triệu",
                "Hiển thị balo nam hoặc balo nữ có rating trên 4.5 sao",
                "Tìm túi xách nữ có giá từ 200k đến 800k",
                "Cho tôi xem giày boots thương hiệu Timberland hoặc Dr.Martens",
                "Tìm kính mát có số lượt đánh giá trên 50 và rating trên 4 sao",
                "Hiển thị dép nam có giá dưới 200k sắp xếp theo giá tăng dần",
                "Tìm vali có kích thước lớn hoặc vừa thương hiệu American Tourister",
                "Cho tôi xem giày cao gót có giá từ 300k đến 1 triệu và rating trên 4 sao"
              ].map((sampleQuery, idx) => (
                <button
                  key={`medium-${idx}`}
                  onClick={() => setQuery(sampleQuery)}
                  className="text-left p-3 bg-yellow-50 hover:bg-yellow-100 border border-yellow-200 rounded-lg transition-colors text-sm"
                  disabled={loading}
                >
                  "{sampleQuery}"
                </button>
              ))}
            </div>
          </div>

          {/* Complex Queries */}
          <div className="mb-4">
            <div className="flex items-center mb-4">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-3"></div>
              <h4 className="font-medium text-red-700 text-lg">Complex Queries</h4>
              <span className="ml-2 text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">Aggregation, JOIN, GROUP BY</span>
            </div>
            <div className="grid grid-cols-1 gap-3">
              {[
                "Tìm giày có giá cao nhất trong danh mục Giày thể thao nam",
                "Hiển thị top 10 balo bán chạy nhất có rating trên 4.5 sao",
                "Đếm số lượng túi xách của từng thương hiệu có giá dưới 500k",
                "Tìm kính mát có rating cao nhất trong khoảng giá từ 200k đến 1 triệu",
                "Hiển thị top 5 thương hiệu có nhiều giày dép nhất",
                "Tìm balo có số lượt đánh giá nhiều nhất trong danh mục Balo laptop",
                "Hiển thị giá trung bình của túi xách theo từng thương hiệu"
              ].map((sampleQuery, idx) => (
                <button
                  key={`complex-${idx}`}
                  onClick={() => setQuery(sampleQuery)}
                  className="text-left p-3 bg-red-50 hover:bg-red-100 border border-red-200 rounded-lg transition-colors text-sm"
                  disabled={loading}
                >
                  "{sampleQuery}"
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}
