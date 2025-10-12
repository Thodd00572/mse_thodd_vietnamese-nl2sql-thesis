import { useState, useEffect } from 'react'
import Layout from '../components/Layout'
import { Search, Clock, Database, AlertCircle, CheckCircle, Download, Cloud, Settings } from 'lucide-react'
import Link from 'next/link'
import api from '../utils/api'
import dbApi from '../utils/dbApi'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedPipeline, setSelectedPipeline] = useState('all')
  const [colabStatus, setColabStatus] = useState(null)
  const [processLogs, setProcessLogs] = useState([])
  const [showProcessView, setShowProcessView] = useState(false)

  useEffect(() => {
    fetchColabStatus()
  }, [])

  const fetchColabStatus = async () => {
    try {
      // Try backend API first
      const response = await api.get('/config/colab/status')
      setColabStatus(response.data.status)
    } catch (error) {
      console.error('Failed to fetch Colab status from backend:', error)
      
      // Try connecting directly to Colab endpoint as fallback
      try {
        const colabResponse = await fetch('https://abnormally-direct-rhino.ngrok-free.app/health', {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          mode: 'cors'
        })
        
        if (colabResponse.ok) {
          const healthData = await colabResponse.json()
          setColabStatus({
            pipeline1_healthy: healthData.pipelines?.P1 || false,
            pipeline2_healthy: healthData.pipelines?.P2 || false,
            pipeline3_healthy: healthData.pipelines?.P3 || false,
            colab_status: healthData.status === 'healthy' ? 'connected' : 'disconnected'
          })
          console.log('Successfully connected to Colab directly:', healthData)
        } else {
          // Set default disconnected status
          setColabStatus({
            pipeline1_healthy: false,
            pipeline2_healthy: false,
            pipeline3_healthy: false,
            colab_status: 'disconnected'
          })
        }
      } catch (directError) {
        console.error('Failed to connect to Colab directly:', directError)
        // Set default disconnected status
        setColabStatus({
          pipeline1_healthy: false,
          pipeline2_healthy: false,
          pipeline3_healthy: false,
          colab_status: 'disconnected'
        })
      }
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
    
    addProcessLog('Starting search process...', 'info')
    addProcessLog(`Query: "${query.trim()}"`, 'info')
    addProcessLog(`Pipeline: ${selectedPipeline}`, 'info')
    
    try {
      const queryText = query.trim()
      const results = {}
      
      // Call individual pipeline endpoints based on selection
      if (selectedPipeline === 'all' || selectedPipeline === 'p1') {
        addProcessLog('Calling P1 (mT5) endpoint...', 'info')
        try {
          const p1Response = await api.post('/p1/generate', { query: queryText })
          results.pipeline1_result = p1Response.data
          addProcessLog('P1 response received', 'success')
        } catch (err) {
          addProcessLog('P1 failed: ' + err.message, 'error')
          results.pipeline1_result = { success: false, error: err.message }
        }
      }
      
      if (selectedPipeline === 'all' || selectedPipeline === 'p2') {
        addProcessLog('Calling P2 (SQLCoder) endpoint...', 'info')
        try {
          const p2Response = await api.post('/p2/generate', { query: queryText })
          results.pipeline2_result = p2Response.data
          addProcessLog('P2 response received', 'success')
        } catch (err) {
          addProcessLog('P2 failed: ' + err.message, 'error')
          results.pipeline2_result = { success: false, error: err.message }
        }
      }
      
      if (selectedPipeline === 'all' || selectedPipeline === 'p3') {
        addProcessLog('Calling P3 (Vanna AI) endpoint...', 'info')
        try {
          const p3Response = await api.post('/p3/generate', { query: queryText })
          results.pipeline3_result = p3Response.data
          addProcessLog('P3 response received', 'success')
        } catch (err) {
          addProcessLog('P3 failed: ' + err.message, 'error')
          results.pipeline3_result = { success: false, error: err.message }
        }
      }
      
      addProcessLog('Processing results...', 'info')
      setResults(results)
      
      // Check for Colab connection errors
      const hasColabError = (results.pipeline1_result?.requires_colab) || 
                           (results.pipeline2_result?.requires_colab) ||
                           (results.pipeline3_result?.requires_colab)
      
      if (hasColabError) {
        addProcessLog('WARNING: Colab server connection required!', 'error')
      }
      
      // Log pipeline results with detailed metrics
      if (results.pipeline1_result) {
        const p1 = results.pipeline1_result
        addProcessLog(`Pipeline 1: ${p1.success ? 'SUCCESS' : 'FAILED'}`, p1.success ? 'success' : 'error')
        addProcessLog(`Pipeline 1 Execution Time: ${(p1.execution_time * 1000).toFixed(2)}ms`, 'info')
        if (p1.sql_query) {
          addProcessLog(`Pipeline 1 SQL: ${p1.sql_query}`, 'info')
        }
        if (p1.error) {
          addProcessLog(`Pipeline 1 Error: ${p1.error}`, 'error')
        }
        if (p1.requires_colab) {
          addProcessLog('Pipeline 1 requires Colab connection', 'error')
        }
      }
      
      if (results.pipeline2_result) {
        const p2 = results.pipeline2_result
        addProcessLog(`Pipeline 2: ${p2.success ? 'SUCCESS' : 'FAILED'}`, p2.success ? 'success' : 'error')
        addProcessLog(`Pipeline 2 Total Time: ${(p2.execution_time * 1000).toFixed(2)}ms`, 'info')
        if (p2.sql_query) {
          addProcessLog(`Pipeline 2 SQL: ${p2.sql_query}`, 'info')
        }
        if (p2.error) {
          addProcessLog(`Pipeline 2 Error: ${p2.error}`, 'error')
        }
        if (p2.requires_colab) {
          addProcessLog('Pipeline 2 requires Colab connection', 'error')
        }
      }
      
      // Handle Pipeline 3 (Vanna AI RAG) results
      if (results.pipeline3_result) {
        const p3 = results.pipeline3_result
        addProcessLog(`Pipeline 3 (Vanna AI): ${p3.success ? 'SUCCESS' : 'FAILED'}`, p3.success ? 'success' : 'error')
        addProcessLog(`Pipeline 3 Total Time: ${(p3.execution_time * 1000).toFixed(2)}ms`, 'info')
        if (p3.sql_query) {
          addProcessLog(`Pipeline 3 SQL: ${p3.sql_query}`, 'info')
        }
        if (p3.error) {
          addProcessLog(`Pipeline 3 Error: ${p3.error}`, 'error')
        }
        if (p3.requires_colab) {
          addProcessLog('Pipeline 3 requires Colab connection', 'error')
        }
      }
      
      // Execute SQL queries against local database to get actual results
      addProcessLog('Executing queries against local database...', 'info')
      
      if (results.pipeline1_result?.sql_query) {
        try {
          const dbResponse = await dbApi.post('/api/database/query', { query: results.pipeline1_result.sql_query })
          results.pipeline1_result.results = dbResponse.data.results
          results.pipeline1_result.rowCount = dbResponse.data.results?.length || 0
          addProcessLog(`P1: Retrieved ${results.pipeline1_result.rowCount} rows`, 'success')
        } catch (err) {
          addProcessLog(`P1 DB execution failed: ${err.message}`, 'error')
          results.pipeline1_result.rowCount = 0
        }
      }
      
      if (results.pipeline2_result?.sql_query) {
        try {
          const dbResponse = await dbApi.post('/api/database/query', { query: results.pipeline2_result.sql_query })
          results.pipeline2_result.results = dbResponse.data.results
          results.pipeline2_result.rowCount = dbResponse.data.results?.length || 0
          addProcessLog(`P2: Retrieved ${results.pipeline2_result.rowCount} rows`, 'success')
        } catch (err) {
          addProcessLog(`P2 DB execution failed: ${err.message}`, 'error')
          results.pipeline2_result.rowCount = 0
        }
      }
      
      if (results.pipeline3_result?.sql_query) {
        try {
          const dbResponse = await dbApi.post('/api/database/query', { query: results.pipeline3_result.sql_query })
          results.pipeline3_result.results = dbResponse.data.results
          results.pipeline3_result.rowCount = dbResponse.data.results?.length || 0
          addProcessLog(`P3: Retrieved ${results.pipeline3_result.rowCount} rows`, 'success')
        } catch (err) {
          addProcessLog(`P3 DB execution failed: ${err.message}`, 'error')
          results.pipeline3_result.rowCount = 0
        }
      }
      
      setResults(results) // Update results with DB query results
      
      if (!hasColabError) {
        addProcessLog('Search completed successfully!', 'success')
      }
      
    } catch (error) {
      console.error('Search error:', error)
      addProcessLog(`API Error: ${error.message}`, 'error')
      
      if (error.response?.data?.detail) {
        addProcessLog(`Error Details: ${error.response.data.detail}`, 'error')
      }
      
      if (error.response?.status) {
        addProcessLog(`HTTP Status: ${error.response.status}`, 'error')
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

  const LocalModelResult = ({ result }) => {
    return (
      <div className="card border-l-4 border-l-purple-500">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
            Local Model
          </h3>
          <div className="flex items-center space-x-2">
            {result.success ? (
              <CheckCircle className="w-5 h-5 text-green-500" />
            ) : (
              <AlertCircle className="w-5 h-5 text-red-500" />
            )}
            <span className="text-sm font-medium text-gray-600">
              {(result.execution_time || 0).toFixed(2)}ms
            </span>
          </div>
        </div>

        {result.sql_query && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 mb-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Generated SQL:</p>
            <code className="text-sm bg-white p-2 rounded border block overflow-x-auto">
              {result.sql_query}
            </code>
          </div>
        )}

        {/* Local Model Metrics */}
        {result.model_used && (
          <div className="bg-white border border-gray-200 rounded-lg p-3 mb-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Model Information:</p>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-gray-600">Model:</span>
                <span className="ml-1 font-mono">{result.model_used}</span>
              </div>
              <div>
                <span className="text-gray-600">Processing Time:</span>
                <span className="ml-1 font-mono text-purple-600">
                  {(result.execution_time || 0).toFixed(2)}ms
                </span>
              </div>
            </div>
          </div>
        )}

        {result.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <p className="text-sm text-red-700">
              <AlertCircle className="w-4 h-4 inline mr-1" />
              {result.error}
            </p>
          </div>
        )}

        {result.results && result.results.length > 0 && (() => {
          // Detect if results are products or aggregated data
          const firstRow = result.results[0];
          const isProductData = firstRow && (firstRow.product_id || firstRow.name);
          const isAggregatedData = !isProductData && firstRow && Object.keys(firstRow).length > 0;

          if (isAggregatedData) {
            // Display aggregated results as a table
            const columns = Object.keys(firstRow);
            return (
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-700 mb-3">
                  Results ({result.results.length} rows):
                </p>
                <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                  <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          {columns.map((col) => (
                            <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                              {col.replace(/_/g, ' ')}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {result.results.map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            {columns.map((col) => (
                              <td key={col} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                {typeof row[col] === 'number' 
                                  ? (col.includes('price') || col.includes('avg') 
                                      ? formatPrice(row[col]) 
                                      : row[col].toLocaleString())
                                  : row[col] || '-'}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            );
          } else {
            // Display product results as cards
            return (
              <div>
                <p className="text-sm font-medium text-gray-700 mb-3">
                  Results ({result.results.length} products):
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-[600px] overflow-y-auto">
                  {result.results.map((product, idx) => (
                    <div key={idx} className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-lg transition-all duration-200 flex flex-col h-full">
                      <div className="flex flex-col h-full">
                        {/* Product Name */}
                        <h4 className="font-semibold text-gray-900 text-sm line-clamp-2 mb-2">
                          {product.name}
                        </h4>
                        
                        {/* Brand Name */}
                        {product.brand && product.brand !== 'Unknown' && (
                          <div className="mb-2">
                            <span className="text-xs font-medium text-blue-600">🏷️ {product.brand}</span>
                          </div>
                        )}
                        
                        {/* Price - Prominent */}
                        {product.price ? (
                          <div className="text-lg font-bold text-green-600 mb-2">
                            {formatPrice(product.price)}
                          </div>
                        ) : (
                          <div className="text-sm text-gray-400 mb-2">Price not available</div>
                        )}
                        
                        {/* Category */}
                        <div className="mb-2">
                          <span className="inline-block px-2 py-1 text-xs font-medium bg-purple-100 text-purple-700 rounded">
                            📂 {product.category || 'Uncategorized'}
                          </span>
                        </div>
                        
                        {/* Description */}
                        <p className="text-xs text-gray-600 line-clamp-2 mb-3 flex-grow">
                          {product.description || 'No description available'}
                        </p>
                        
                        {/* Rating */}
                        <div className="mt-auto pt-2 border-t border-gray-100">
                          {product.rating ? (
                            <div className="flex items-center gap-1">
                              <span className="text-yellow-500">⭐</span>
                              <span className="text-sm font-medium">{product.rating.toFixed(1)}</span>
                              {product.review_count && (
                                <span className="text-xs text-gray-500">({product.review_count})</span>
                              )}
                            </div>
                          ) : (
                            <span className="text-xs text-gray-400">No ratings</span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          }
        })()}

        {/* Debug Information */}
        {result.debug_info && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 mt-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Debug Information:</p>
            <div className="text-xs space-y-1">
              {result.debug_info.raw_output && (
                <div>
                  <span className="text-gray-600">Raw Output:</span>
                  <span className="ml-1 font-mono text-gray-800">{result.debug_info.raw_output}</span>
                </div>
              )}
              {result.debug_info.cleaned_sql && (
                <div>
                  <span className="text-gray-600">Cleaned SQL:</span>
                  <span className="ml-1 font-mono text-gray-800">{result.debug_info.cleaned_sql}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    )
  }

  const PipelineResult = ({ result, pipelineNumber }) => {
    if (!result) return null

    const getColors = (num) => {
      switch(num) {
        case 1: return { bg: 'bg-blue-50 border-blue-200', accent: 'text-blue-700', dot: 'bg-blue-500' }
        case 2: return { bg: 'bg-green-50 border-green-200', accent: 'text-green-700', dot: 'bg-green-500' }
        case 3: return { bg: 'bg-purple-50 border-purple-200', accent: 'text-purple-700', dot: 'bg-purple-500' }
        default: return { bg: 'bg-gray-50 border-gray-200', accent: 'text-gray-700', dot: 'bg-gray-500' }
      }
    }
    
    const colors = getColors(pipelineNumber)

    return (
      <div className={`${colors.bg} border rounded-xl p-6 space-y-4`}>
        <div className="flex items-center justify-between">
          <h3 className={`font-semibold ${colors.accent} text-lg flex items-center`}>
            <div className={`w-3 h-3 ${colors.dot} rounded-full mr-2`}></div>
            {result.pipeline_name || `Pipeline ${pipelineNumber}`}
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


        {result.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <p className="text-sm text-red-700">
              <AlertCircle className="w-4 h-4 inline mr-1" />
              {result.error}
            </p>
          </div>
        )}

        {result.results && result.results.length > 0 && (() => {
          // Detect if results are products or aggregated data
          const firstRow = result.results[0];
          const isProductData = firstRow && (firstRow.product_id || firstRow.name);
          const isAggregatedData = !isProductData && firstRow && Object.keys(firstRow).length > 0;

          if (isAggregatedData) {
            // Display aggregated results as a table
            const columns = Object.keys(firstRow);
            return (
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-700 mb-3">
                  Results ({result.results.length} rows):
                </p>
                <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                  <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          {columns.map((col) => (
                            <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                              {col.replace(/_/g, ' ')}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {result.results.map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            {columns.map((col) => (
                              <td key={col} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                {typeof row[col] === 'number' 
                                  ? (col.includes('price') || col.includes('avg') 
                                      ? formatPrice(row[col]) 
                                      : row[col].toLocaleString())
                                  : row[col] || '-'}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            );
          } else {
            // Display product results as cards
            return (
              <div>
                <p className="text-sm font-medium text-gray-700 mb-3">
                  Results ({result.results.length} products):
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {result.results.map((product, idx) => (
                    <div key={idx} className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-lg transition-all duration-200 flex flex-col h-full">
                      <div className="flex flex-col h-full">
                        {/* Product Name */}
                        <h4 className="font-semibold text-gray-900 text-sm line-clamp-2 mb-2">
                          {product.name}
                        </h4>
                        
                        {/* Brand Name */}
                        {product.brand && product.brand !== 'Unknown' && (
                          <div className="mb-2">
                            <span className="text-xs font-medium text-blue-600">🏷️ {product.brand}</span>
                          </div>
                        )}
                        
                        {/* Price - Prominent */}
                        {product.price ? (
                          <div className="text-lg font-bold text-green-600 mb-2">
                            {formatPrice(product.price)}
                          </div>
                        ) : (
                          <div className="text-sm text-gray-400 mb-2">Price not available</div>
                        )}
                        
                        {/* Category */}
                        <div className="mb-2">
                          <span className="inline-block px-2 py-1 text-xs font-medium bg-purple-100 text-purple-700 rounded">
                            📂 {product.category || 'Uncategorized'}
                          </span>
                        </div>
                        
                        {/* Description */}
                        <p className="text-xs text-gray-600 line-clamp-2 mb-3 flex-grow">
                          {product.description || 'No description available'}
                        </p>
                        
                        {/* Rating */}
                        <div className="mt-auto pt-2 border-t border-gray-100">
                          {product.rating ? (
                            <div className="flex items-center gap-1">
                              <span className="text-yellow-500">⭐</span>
                              <span className="text-sm font-medium">{product.rating.toFixed(1)}</span>
                              {product.review_count && (
                                <span className="text-xs text-gray-500">({product.review_count})</span>
                              )}
                            </div>
                          ) : (
                            <span className="text-xs text-gray-400">No ratings</span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          }
        })()}

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
                Compare three Vietnamese NL2SQL pipelines: P1 (mT5 Zero-Shot), P2 (SQLCoder Zero-Shot), P3 (Vanna AI RAG)
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
                    <div className="flex items-center space-x-1">
                      <div className={`w-2 h-2 rounded-full ${colabStatus.pipeline3_healthy ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span className="text-xs text-gray-600">P3</span>
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
          {colabStatus && (!colabStatus.pipeline1_healthy || !colabStatus.pipeline2_healthy || !colabStatus.pipeline3_healthy) && (
            <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-yellow-500 mr-2" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-yellow-800">
                    ⚠️ Some pipelines require Colab connection
                  </p>
                  <p className="text-xs text-yellow-700 mt-1">
                    {!colabStatus.pipeline1_healthy && "P1 (mT5 Zero-Shot): Not connected. "}
                    {!colabStatus.pipeline2_healthy && "P2 (SQLCoder Zero-Shot): Not connected. "}
                    {!colabStatus.pipeline3_healthy && "P3 (Vanna AI RAG): Not connected. "}
                    Configure Colab endpoints to enable full functionality.
                  </p>
                </div>
                <Link href="/config" className="btn-primary text-xs px-3 py-1">
                  Configure
                </Link>
              </div>
            </div>
          )}
          
          {/* Additional error banner for when results show Colab requirement */}
          {results && (results.pipeline1_result?.requires_colab || results.pipeline2_result?.requires_colab || results.pipeline3_result?.requires_colab) && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-red-800">
                    🚫 Pipeline Connection Required
                  </p>
                  <p className="text-xs text-red-700 mt-1">
                    Selected pipelines require Colab API endpoints. Configure the pipeline URLs in the Config page to enable Vietnamese NL2SQL processing.
                  </p>
                </div>
                <Link href="/config" className="btn-primary text-xs px-3 py-1">
                  Configure Now
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
              <div className="grid grid-cols-2 gap-4">
                {[
                  { value: 'all', label: 'All Pipelines', description: 'Compare all three approaches' },
                  { value: 'pipeline1', label: 'P1: mT5 Zero-Shot', description: 'EM: 16%, EX: 33%, 304ms' },
                  { value: 'pipeline2', label: 'P2: SQLCoder Zero-Shot', description: 'EM: 18%, EX: 22%, 1,763ms' },
                  { value: 'pipeline3', label: 'P3: Vanna AI RAG', description: 'EM: 43%, EX: 76%, 1,779ms (Best)' }
                ].map((option) => (
                  <label key={option.value} className="flex items-start p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
                    <input
                      type="radio"
                      name="pipeline"
                      value={option.value}
                      checked={selectedPipeline === option.value}
                      onChange={(e) => setSelectedPipeline(e.target.value)}
                      className="mt-1 mr-3"
                      disabled={loading}
                    />
                    <div>
                      <span className="text-sm font-medium text-gray-900">{option.label}</span>
                      <p className="text-xs text-gray-500 mt-1">{option.description}</p>
                    </div>
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

            {/* Pipeline Results - Horizontal Layout */}
            <div className="space-y-4">
              {results.pipeline1_result && (
                <PipelineResult result={results.pipeline1_result} pipelineNumber={1} />
              )}
              {results.pipeline2_result && (
                <PipelineResult result={results.pipeline2_result} pipelineNumber={2} />
              )}
              {results.pipeline3_result && (
                <PipelineResult result={results.pipeline3_result} pipelineNumber={3} />
              )}
            </div>

            {/* Comparison Summary */}
            {(results.pipeline1_result || results.pipeline2_result || results.pipeline3_result) && (
              <div className="card bg-gradient-to-r from-blue-50 via-green-50 to-purple-50">
                <h3 className="font-semibold text-gray-900 mb-4">Pipeline Comparison</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Success Status</p>
                    <div className="flex justify-center space-x-2 mt-2">
                      {results.pipeline1_result && (
                        <span className={results.pipeline1_result.success ? 'status-success' : 'status-error'}>
                          P1
                        </span>
                      )}
                      {results.pipeline2_result && (
                        <span className={results.pipeline2_result.success ? 'status-success' : 'status-error'}>
                          P2
                        </span>
                      )}
                      {results.pipeline3_result && (
                        <span className={results.pipeline3_result.success ? 'status-success' : 'status-error'}>
                          P3
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Execution Times</p>
                    <div className="text-xs space-y-1 mt-2">
                      {results.pipeline1_result && (
                        <div>P1: {(results.pipeline1_result.execution_time * 1000).toFixed(0)}ms</div>
                      )}
                      {results.pipeline2_result && (
                        <div>P2: {(results.pipeline2_result.execution_time * 1000).toFixed(0)}ms</div>
                      )}
                      {results.pipeline3_result && (
                        <div>P3: {(results.pipeline3_result.execution_time * 1000).toFixed(0)}ms</div>
                      )}
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Result Counts</p>
                    <div className="text-xs space-y-1 mt-2">
                      {results.pipeline1_result && (
                        <div>P1: {results.pipeline1_result.rowCount || 0}</div>
                      )}
                      {results.pipeline2_result && (
                        <div>P2: {results.pipeline2_result.rowCount || 0}</div>
                      )}
                      {results.pipeline3_result && (
                        <div>P3: {results.pipeline3_result.rowCount || 0}</div>
                      )}
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Best Performer</p>
                    <div className="text-sm font-semibold mt-2">
                      {(() => {
                        const pipelines = []
                        if (results.pipeline1_result?.success) pipelines.push({ name: 'P1', time: results.pipeline1_result.execution_time, results: results.pipeline1_result.rowCount || 0 })
                        if (results.pipeline2_result?.success) pipelines.push({ name: 'P2', time: results.pipeline2_result.execution_time, results: results.pipeline2_result.rowCount || 0 })
                        if (results.pipeline3_result?.success) pipelines.push({ name: 'P3', time: results.pipeline3_result.execution_time, results: results.pipeline3_result.rowCount || 0 })
                        
                        if (pipelines.length === 0) return 'None'
                        
                        // Find fastest with results
                        const withResults = pipelines.filter(p => p.results > 0)
                        if (withResults.length === 0) return pipelines.sort((a, b) => a.time - b.time)[0].name
                        
                        return withResults.sort((a, b) => a.time - b.time)[0].name
                      })()}
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
              <span className="ml-2 text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">From eval_data.jsonl (index 0-99) - High Success Rate</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {[
                "Tìm áo thun",
                "Hiển thị giày",
                "Xem túi xách",
                "Tìm ví",
                "Hiển thị dép",
                "Xem nón",
                "Tìm đồng hồ",
                "Tìm quần",
                "Hiển thị váy"
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
              <span className="ml-2 text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full">From eval_data.jsonl (index 100-199) - Requires JOINs</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                "Sản phẩm theo thương hiệu",
                "Giá trung bình theo danh mục",
                "Sản phẩm có đánh giá cao",
                "Thương hiệu Nike",
                "Sản phẩm giá dưới 500k",
                "Sản phẩm thương hiệu Adidas"
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
              <span className="ml-2 text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">From eval_data.jsonl (index 200-299) - Multi-table JOINs</span>
            </div>
            <div className="grid grid-cols-1 gap-3">
              {[
                "Top 10 sản phẩm đánh giá cao nhất có giá dưới 1 triệu",
                "Phân tích thị phần thương hiệu",
                "Top 5 sản phẩm bán chạy nhất trong danh mục Phụ kiện thời trang",
                "Top 10 sản phẩm bán chạy nhất trong danh mục Giày dép nam"
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
