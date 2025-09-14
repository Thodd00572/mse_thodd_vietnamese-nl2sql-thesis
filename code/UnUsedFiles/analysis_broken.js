import { useState, useEffect } from 'react'
import Layout from '../components/Layout'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area } from 'recharts'
import { Download, RefreshCw, TrendingUp, Clock, CheckCircle, AlertCircle, Zap, Target, Activity, Database, FileText } from 'lucide-react'
import api from '../utils/api'

function AnalysisPage() {
  const [analysisData, setAnalysisData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [exportLoading, setExportLoading] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchAnalysisData = async () => {
    setLoading(true)
    try {
      // Load sample analysis results from local file
      const response = await fetch('/data/analysis_results.json')
      const data = await response.json()
      setAnalysisData(data)
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (error) {
      console.error('Analysis data fetch error:', error)
      // Fallback to API if local file not available
      try {
        const apiResponse = await api.get('/analyze')
        setAnalysisData(apiResponse.data)
        setLastUpdated(new Date().toLocaleTimeString())
      } catch (apiError) {
        console.error('API fetch error:', apiError)
      }
    } finally {
      setLoading(false)
    }
  }

  const exportData = async (dataType, format = 'csv') => {
    setExportLoading(true)
    try {
      const response = await api.get(`/export/${dataType}?format=${format}`)
      
      // Create download link
      const downloadUrl = `https://abnormally-direct-rhino.ngrok-free.app${response.data.download_url}`
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = response.data.filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
    } catch (error) {
      console.error('Export error:', error)
    } finally {
      setExportLoading(false)
    }
  }

  const resetExperiment = async () => {
    try {
      await api.post('/reset')
      setAnalysisData(null)
      alert('Analysis data reset successfully!')
    } catch (error) {
      console.error('Reset error:', error)
    }
  }

  useEffect(() => {
    fetchAnalysisData()
    
    // Set up live updates every 30 seconds
    const interval = setInterval(fetchAnalysisData, 30000)
    return () => clearInterval(interval)
  }, [])

  const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444']

  if (loading && !analysisData) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="loading-spinner"></div>
          <span className="ml-2">Loading 150-query analysis results...</span>
        </div>
      </Layout>
    )
  }

  const comparisonData = analysisData ? [
    {
      name: 'Pipeline 1 (PhoBERT-SQL)',
      'Success Rate (%)': analysisData.overall_statistics.pipeline1_results.success_rate,
      'Avg Latency (ms)': analysisData.overall_statistics.pipeline1_results.avg_execution_time_ms,
      'Success Count': analysisData.overall_statistics.pipeline1_results.successful,
      'Error Count': analysisData.overall_statistics.pipeline1_results.failed,
      'Total Results': analysisData.overall_statistics.pipeline1_results.total_results_returned
    },
    {
      name: 'Pipeline 2 (Vi→En→SQL)', 
      'Success Rate (%)': analysisData.overall_statistics.pipeline2_results.success_rate,
      'Avg Latency (ms)': analysisData.overall_statistics.pipeline2_results.avg_execution_time_ms,
      'Success Count': analysisData.overall_statistics.pipeline2_results.successful,
      'Error Count': analysisData.overall_statistics.pipeline2_results.failed,
      'Total Results': analysisData.overall_statistics.pipeline2_results.total_results_returned
    }
  ] : []

  const performanceData = analysisData ? [
    { name: 'Success Rate (%)', Pipeline1: analysisData.overall_statistics.pipeline1_results.success_rate, Pipeline2: analysisData.overall_statistics.pipeline2_results.success_rate },
    { name: 'Avg Latency (ms)', Pipeline1: analysisData.overall_statistics.pipeline1_results.avg_execution_time_ms, Pipeline2: analysisData.overall_statistics.pipeline2_results.avg_execution_time_ms },
    { name: 'Results Returned', Pipeline1: analysisData.overall_statistics.pipeline1_results.total_results_returned, Pipeline2: analysisData.overall_statistics.pipeline2_results.total_results_returned }
  ] : []

  const complexityData = analysisData ? [
    { name: 'Simple', Pipeline1: analysisData.complexity_breakdown.simple_queries.pipeline1.success_rate, Pipeline2: analysisData.complexity_breakdown.simple_queries.pipeline2.success_rate },
    { name: 'Medium', Pipeline1: analysisData.complexity_breakdown.medium_queries.pipeline1.success_rate, Pipeline2: analysisData.complexity_breakdown.medium_queries.pipeline2.success_rate },
    { name: 'Complex', Pipeline1: analysisData.complexity_breakdown.complex_queries.pipeline1.success_rate, Pipeline2: analysisData.complexity_breakdown.complex_queries.pipeline2.success_rate }
  ] : []

  const timelineData = analysisData ? analysisData.performance_trends.execution_timeline : []

  return (
    <Layout>
      <div className="px-4 sm:px-0">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Pipeline Analysis
            </h1>
            <p className="mt-2 text-gray-600">
              Performance metrics and comparison between translation pipelines
            </p>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={fetchAnalysisData}
              disabled={loading}
              className="btn-secondary flex items-center"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button
              onClick={resetExperiment}
              className="btn-secondary text-red-600 hover:bg-red-50"
            >
              Reset Data
            </button>
          </div>
        </div>

        {!analysisData ? (
          <div className="card text-center py-12">
            <TrendingUp className="w-12 h-12 mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Analysis Data Available</h3>
            <p className="text-gray-600 mb-4">
              Execute the 150 Vietnamese queries via Colab API to generate comprehensive analysis data
            </p>
            <button
              onClick={() => window.location.href = '/'}
              className="btn-primary"
            >
              Go to Search
            </button>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Live Status Banner */}
            <div className="bg-gradient-to-r from-blue-50 to-green-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Activity className="w-5 h-5 text-green-500 mr-2" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      150 Vietnamese Query Analysis - Live Results
                    </p>
                    <p className="text-xs text-gray-600">
                      Last updated: {lastUpdated} | Duration: {analysisData.analysis_metadata.test_duration_minutes} minutes | Source: {analysisData.analysis_metadata.query_source}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-xs text-green-600 font-medium">Live Data</span>
                </div>
              </div>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
              <div className="metric-card">
                <div className="flex items-center">
                  <div className="p-2 bg-primary-100 rounded-lg">
                    <Database className="w-6 h-6 text-primary-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Total Queries</p>
                    <p className="text-2xl font-bold text-gray-900">{analysisData.analysis_metadata.total_queries}</p>
                  </div>
                </div>
              </div>

              <div className="metric-card">
                <div className="flex items-center">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <Zap className="w-6 h-6 text-blue-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Faster Pipeline</p>
                    <p className="text-lg font-bold text-gray-900">
                      {analysisData.overall_statistics.comparison.pipeline1_faster_count > analysisData.overall_statistics.comparison.pipeline2_faster_count ? 'Pipeline 1' : 'Pipeline 2'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {analysisData.overall_statistics.comparison.avg_time_difference_ms.toFixed(1)}ms avg difference
                    </p>
                  </div>
                </div>
              </div>

              <div className="metric-card">
                <div className="flex items-center">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <CheckCircle className="w-6 h-6 text-green-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">More Accurate</p>
                    <p className="text-lg font-bold text-gray-900">
                      {analysisData.overall_statistics.pipeline1_results.success_rate > analysisData.overall_statistics.pipeline2_results.success_rate ? 'Pipeline 1' : 'Pipeline 2'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {analysisData.overall_statistics.comparison.accuracy_difference.toFixed(1)}% difference
                    </p>
                  </div>
                </div>
              </div>

              <div className="metric-card">
                <div className="flex items-center">
                  <div className="p-2 bg-yellow-100 rounded-lg">
                    <Clock className="w-6 h-6 text-yellow-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Test Duration</p>
                    <p className="text-lg font-bold text-gray-900">
                      {analysisData.analysis_metadata.test_duration_minutes}m
                    </p>
                  </div>
                </div>
              </div>

              <div className="metric-card">
                <div className="flex items-center">
                  <div className="p-2 bg-purple-100 rounded-lg">
                    <Target className="w-6 h-6 text-purple-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Overall Success</p>
                    <p className="text-lg font-bold text-gray-900">
                      {((analysisData.overall_statistics.pipeline1_results.successful + analysisData.overall_statistics.pipeline2_results.successful) / (analysisData.analysis_metadata.total_queries * 2) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Research Insights */}
            <div className="card bg-gradient-to-r from-blue-50 to-green-50">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Research Insights from 150-Query Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Key Findings</h4>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Pipeline 1 achieved {analysisData.overall_statistics.pipeline1_results.success_rate}% success rate vs Pipeline 2's {analysisData.overall_statistics.pipeline2_results.success_rate}%</li>
                    <li>• Pipeline 1 is faster by {analysisData.overall_statistics.comparison.avg_time_difference_ms.toFixed(1)}ms on average</li>
                    <li>• Complex queries show {analysisData.complexity_breakdown.complex_queries.pipeline1.success_rate - analysisData.complexity_breakdown.complex_queries.pipeline2.success_rate}% accuracy difference</li>
                    <li>• Translation step adds ~{analysisData.overall_statistics.pipeline2_results.avg_translation_time_ms.toFixed(0)}ms overhead in Pipeline 2</li>
                    <li>• Pipeline 1 returned {analysisData.overall_statistics.pipeline1_results.total_results_returned} total results vs {analysisData.overall_statistics.pipeline2_results.total_results_returned}</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Recommendations</h4>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Use Pipeline 1 for real-time applications requiring low latency</li>
                    <li>• Pipeline 2 may benefit from translation model optimization</li>
                    <li>• Focus on improving complex query handling in both pipelines</li>
                    <li>• Consider hybrid approach: Pipeline 1 for simple queries, Pipeline 2 for complex ones</li>
                    <li>• Investigate {analysisData.error_analysis.pipeline1_errors[0].error_type.toLowerCase()} errors in Pipeline 1</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Detailed Metrics Table */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Pipeline 1 Metrics */}
              <div className="card">
                <h3 className="text-lg font-semibold text-blue-700 mb-4">
                  Pipeline 1: Vietnamese → PhoBERT-SQL
                </h3>
                <div className="space-y-3">
                  {[
                    { label: 'Success Count', value: analysisData.overall_statistics.pipeline1_results.successful },
                    { label: 'Error Count', value: analysisData.overall_statistics.pipeline1_results.failed },
                    { label: 'Success Rate', value: `${analysisData.overall_statistics.pipeline1_results.success_rate}%` },
                    { label: 'Average Latency', value: `${analysisData.overall_statistics.pipeline1_results.avg_execution_time_ms.toFixed(1)}ms` },
                    { label: 'Min Latency', value: `${analysisData.overall_statistics.pipeline1_results.min_execution_time_ms.toFixed(1)}ms` },
                    { label: 'Max Latency', value: `${analysisData.overall_statistics.pipeline1_results.max_execution_time_ms.toFixed(1)}ms` },
                    { label: 'Total Results', value: analysisData.overall_statistics.pipeline1_results.total_results_returned },
                    { label: 'Avg SQL Length', value: `${analysisData.overall_statistics.pipeline1_results.avg_sql_length} chars` }
                  ].map((metric, idx) => (
                    <div key={idx} className="flex justify-between py-2 border-b border-gray-100 last:border-b-0">
                      <span className="text-sm text-gray-600">{metric.label}</span>
                      <span className="text-sm font-medium text-gray-900">{metric.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Pipeline 2 Metrics */}
              <div className="card">
                <h3 className="text-lg font-semibold text-green-700 mb-4">
                  Pipeline 2: Vietnamese → PhoBERT → SQLCoder
                </h3>
                <div className="space-y-3">
                  {[
                    { label: 'Success Count', value: analysisData.overall_statistics.pipeline2_results.successful },
                    { label: 'Error Count', value: analysisData.overall_statistics.pipeline2_results.failed },
                    { label: 'Success Rate', value: `${analysisData.overall_statistics.pipeline2_results.success_rate}%` },
                    { label: 'Average Latency', value: `${analysisData.overall_statistics.pipeline2_results.avg_execution_time_ms.toFixed(1)}ms` },
                    { label: 'Avg Translation Time', value: `${analysisData.overall_statistics.pipeline2_results.avg_translation_time_ms.toFixed(1)}ms` },
                    { label: 'Avg SQL Gen Time', value: `${analysisData.overall_statistics.pipeline2_results.avg_sql_generation_time_ms.toFixed(1)}ms` },
                    { label: 'Total Results', value: analysisData.overall_statistics.pipeline2_results.total_results_returned },
                    { label: 'Avg SQL Length', value: `${analysisData.overall_statistics.pipeline2_results.avg_sql_length} chars` }
                  ].map((metric, idx) => (
                    <div key={idx} className="flex justify-between py-2 border-b border-gray-100 last:border-b-0">
                      <span className="text-sm text-gray-600">{metric.label}</span>
                      <span className="text-sm font-medium text-gray-900">{metric.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Performance Charts */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Performance Analysis Charts</h3>
              
              {/* Execution Timeline */}
              <div className="mb-8">
                <h4 className="text-md font-medium text-gray-800 mb-4">Execution Timeline (Live Updates)</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="minute" label={{ value: 'Minutes', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'Avg Response Time (ms)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="pipeline1_avg_ms" stroke="#3b82f6" strokeWidth={3} dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }} name="Pipeline 1" />
                    <Line type="monotone" dataKey="pipeline2_avg_ms" stroke="#22c55e" strokeWidth={3} dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }} name="Pipeline 2" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Complexity Breakdown */}
              <div className="mb-8">
                <h4 className="text-md font-medium text-gray-800 mb-4">Success Rate by Query Complexity</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={complexityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: 'Success Rate (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Pipeline1" fill="#3b82f6" name="Pipeline 1 (PhoBERT-SQL)" />
                    <Bar dataKey="Pipeline2" fill="#22c55e" name="Pipeline 2 (Vi→En→SQL)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Performance Metrics Comparison */}
              <div>
                <h4 className="text-md font-medium text-gray-800 mb-4">Performance Metrics Comparison</h4>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="Pipeline1" stroke="#3b82f6" strokeWidth={3} />
                    <Line type="monotone" dataKey="Pipeline2" stroke="#22c55e" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Error Analysis */
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Error Analysis</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-blue-700 mb-3">Pipeline 1 Errors</h4>
                  <div className="space-y-3">
                    {analysisData.error_analysis.pipeline1_errors.map((error, idx) => (
                      <div key={idx} className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium text-blue-800">{error.error_type}</span>
                          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">{error.count} errors ({error.percentage}%)</span>
                        </div>
                        <div className="text-xs text-blue-600">
                          Sample: "{error.sample_queries[0]}"
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-green-700 mb-3">Pipeline 2 Errors</h4>
                  <div className="space-y-3">
                    {analysisData.error_analysis.pipeline2_errors.map((error, idx) => (
                      <div key={idx} className="bg-green-50 border border-green-200 rounded-lg p-3">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium text-green-800">{error.error_type}</span>
                          <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">{error.count} errors ({error.percentage}%)</span>
                        </div>
                        <div className="text-xs text-green-600">
                          Sample: "{error.sample_queries[0]}"
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Sample Query Results */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Sample Query Executions</h3>
              <div className="space-y-4">
                {analysisData.query_results_sample.map((sample, idx) => (
                  <div key={idx} className="border border-gray-200 rounded-lg p-4">
                    <div className="mb-3">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-900">Query #{sample.query_id}</h4>
                        <span className={`px-2 py-1 text-xs rounded ${
                          sample.complexity === 'Simple' ? 'bg-green-100 text-green-700' :
                          sample.complexity === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {sample.complexity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 italic">"{sample.vietnamese_query}"</p>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Pipeline 1 Result */}
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-blue-800">Pipeline 1</span>
                          <span className={`text-xs px-2 py-1 rounded ${
                            sample.pipeline1.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                          }`}>
                            {sample.pipeline1.success ? 'SUCCESS' : 'FAILED'}
                          </span>
                        </div>
                        <div className="text-xs text-blue-600 mb-1">
                          Time: {sample.pipeline1.execution_time_ms.toFixed(1)}ms | Results: {sample.pipeline1.results_count}
                        </div>
                        {sample.pipeline1.sql_query && (
                          <div className="text-xs text-gray-600 bg-white p-2 rounded border font-mono">
                            {sample.pipeline1.sql_query.substring(0, 100)}...
                          </div>
                        )}
                        {sample.pipeline1.error && (
                          <div className="text-xs text-red-600 mt-1">{sample.pipeline1.error}</div>
                        )}
                      </div>
                      
                      {/* Pipeline 2 Result */}
                      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-green-800">Pipeline 2</span>
                          <span className={`text-xs px-2 py-1 rounded ${
                            sample.pipeline2.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                          }`}>
                            {sample.pipeline2.success ? 'SUCCESS' : 'FAILED'}
                          </span>
                        </div>
                        <div className="text-xs text-green-600 mb-1">
                          Time: {sample.pipeline2.execution_time_ms.toFixed(1)}ms | Results: {sample.pipeline2.results_count}
                        </div>
                        {sample.pipeline2.english_query && (
                          <div className="text-xs text-gray-600 mb-1">
                            EN: "{sample.pipeline2.english_query}"
                          </div>
                        )}
                        {sample.pipeline2.sql_query && (
                          <div className="text-xs text-gray-600 bg-white p-2 rounded border font-mono">
                            {sample.pipeline2.sql_query.substring(0, 100)}...
                          </div>
                        )}
                        {sample.pipeline2.error && (
                          <div className="text-xs text-red-600 mt-1">{sample.pipeline2.error}</div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Colab Server Status */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Colab Server Status</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-sm text-gray-600">Server URL</p>
                  <p className="text-sm font-mono text-gray-900">
                    {analysisData.analysis_metadata.colab_server_url}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Pipeline 1 Health</p>
                  <p className="text-xl font-bold text-green-600">
                    {analysisData.real_time_status.colab_server_health.pipeline1_healthy ? '✅ Healthy' : '❌ Down'}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Pipeline 2 Health</p>
                  <p className="text-xl font-bold text-green-600">
                    {analysisData.real_time_status.colab_server_health.pipeline2_healthy ? '✅ Healthy' : '❌ Down'}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Last Health Check</p>
                  <p className="text-sm text-gray-900">
                    {new Date(analysisData.real_time_status.colab_server_health.last_health_check).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>



          </div>
        )}
      </div>
    </Layout>
  )
}

export default AnalysisPage
