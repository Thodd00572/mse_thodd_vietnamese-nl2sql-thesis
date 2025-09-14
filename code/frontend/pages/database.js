import { useState, useEffect } from 'react'
import Layout from '../components/Layout'
import { 
  Database, 
  Table, 
  Search, 
  Play, 
  Download, 
  RefreshCw, 
  BarChart3, 
  FileText, 
  Server,
  Eye,
  ChevronLeft,
  ChevronRight,
  GitBranch
} from 'lucide-react'
import api from '../utils/api'
import mermaid from 'mermaid'

export default function DatabaseManagementPage() {
  // State management
  const [activeTab, setActiveTab] = useState('overview')
  const [dbStats, setDbStats] = useState(null)
  const [queryResult, setQueryResult] = useState(null)
  const [sqlQuery, setSqlQuery] = useState('SELECT * FROM products LIMIT 10;')
  const [sampleData, setSampleData] = useState([])
  const [loading, setLoading] = useState(false)
  const [queryLoading, setQueryLoading] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(50)

  // Initialize Mermaid
  useEffect(() => {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      securityLevel: 'loose',
    })
  }, [])

  // Fetch database statistics
  const fetchDatabaseStats = async () => {
    setLoading(true)
    try {
      const response = await api.get('/api/database/stats')
      setDbStats(response.data)
    } catch (error) {
      console.error('Database stats error:', error)
    } finally {
      setLoading(false)
    }
  }

  // Execute SQL query
  const executeQuery = async () => {
    if (!sqlQuery.trim()) return
    
    setQueryLoading(true)
    try {
      const response = await api.post('/api/database/query', { query: sqlQuery })
      setQueryResult(response.data)
    } catch (error) {
      console.error('Query execution error:', error)
      setQueryResult({ error: error.response?.data?.detail || error.message })
    } finally {
      setQueryLoading(false)
    }
  }

  // Fetch sample data
  const fetchSampleData = async (page = 1) => {
    setLoading(true)
    try {
      const response = await api.get('/api/products', {
        params: { page, limit: itemsPerPage }
      })
      setSampleData(response.data.products || [])
      setCurrentPage(page)
    } catch (error) {
      console.error('Sample data error:', error)
    } finally {
      setLoading(false)
    }
  }

  // Load initial data
  useEffect(() => {
    fetchDatabaseStats()
    fetchSampleData()
  }, [])

  // Predefined queries
  const predefinedQueries = [
    {
      name: 'Top 10 Products by Price',
      query: 'SELECT name, price, brand, category FROM products ORDER BY price DESC LIMIT 10;'
    },
    {
      name: 'Category Distribution',
      query: 'SELECT category, COUNT(*) as count FROM products GROUP BY category ORDER BY count DESC LIMIT 20;'
    },
    {
      name: 'Brand Analysis',
      query: 'SELECT brand, COUNT(*) as products, AVG(price) as avg_price FROM products WHERE brand IS NOT NULL GROUP BY brand ORDER BY products DESC LIMIT 15;'
    },
    {
      name: 'Price Range Analysis',
      query: 'SELECT CASE WHEN price < 100000 THEN "Under 100k" WHEN price < 500000 THEN "100k-500k" WHEN price < 1000000 THEN "500k-1M" ELSE "Over 1M" END as price_range, COUNT(*) as count FROM products GROUP BY price_range;'
    }
  ]

  return (
    <Layout>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="py-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Database className="w-8 h-8 text-blue-600 mr-3" />
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">Database Management</h1>
                    <p className="text-sm text-gray-600 mt-1">Vietnamese Tiki Products Database</p>
                  </div>
                </div>
                <button
                  onClick={fetchDatabaseStats}
                  disabled={loading}
                  className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <nav className="flex space-x-8">
              {[
                { id: 'overview', name: 'Overview', icon: BarChart3 },
                { id: 'query', name: 'SQL Query', icon: Play },
                { id: 'data', name: 'Data Browser', icon: Table },
                { id: 'schema', name: 'Schema', icon: FileText }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="w-4 h-4 mr-2" />
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Statistics Cards */}
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  <span className="ml-2 text-gray-600">Loading statistics...</span>
                </div>
              ) : dbStats ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div className="bg-white rounded-lg shadow p-6">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <Server className="h-8 w-8 text-blue-600" />
                        </div>
                        <div className="ml-4">
                          <p className="text-sm font-medium text-gray-500">Total Records</p>
                          <p className="text-2xl font-semibold text-gray-900">
                            {dbStats.totalProducts?.[0]?.count?.toLocaleString() || '0'}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-white rounded-lg shadow p-6">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <Table className="h-8 w-8 text-green-600" />
                        </div>
                        <div className="ml-4">
                          <p className="text-sm font-medium text-gray-500">Categories</p>
                          <p className="text-2xl font-semibold text-gray-900">
                            {dbStats.categoryStats?.length || 0}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-white rounded-lg shadow p-6">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <FileText className="h-8 w-8 text-purple-600" />
                        </div>
                        <div className="ml-4">
                          <p className="text-sm font-medium text-gray-500">Unique Brands</p>
                          <p className="text-2xl font-semibold text-gray-900">
                            {dbStats.brandCount || 0}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-white rounded-lg shadow p-6">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <BarChart3 className="h-8 w-8 text-yellow-600" />
                        </div>
                        <div className="ml-4">
                          <p className="text-sm font-medium text-gray-500">Avg Price</p>
                          <p className="text-2xl font-semibold text-gray-900">
                            {Math.round(dbStats.priceStats?.[0]?.avg_price || 0).toLocaleString()} ₫
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Top Categories */}
                  <div className="bg-white rounded-lg shadow">
                    <div className="px-6 py-4 border-b border-gray-200">
                      <h3 className="text-lg font-medium text-gray-900">Top Categories</h3>
                    </div>
                    <div className="p-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {dbStats.categoryStats?.slice(0, 9).map((category, idx) => (
                          <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                            <span className="text-sm font-medium text-gray-900 truncate">
                              {category.category}
                            </span>
                            <span className="text-sm text-gray-600 bg-white px-2 py-1 rounded">
                              {category.count.toLocaleString()}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Data Sources */}
                  <div className="bg-white rounded-lg shadow">
                    <div className="px-6 py-4 border-b border-gray-200">
                      <h3 className="text-lg font-medium text-gray-900">Data Sources</h3>
                    </div>
                    <div className="p-6">
                      <div className="space-y-3">
                        {dbStats.fileStats?.map((file, idx) => (
                          <div key={idx} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                            <div className="flex items-center">
                              <FileText className="w-5 h-5 text-gray-400 mr-3" />
                              <span className="font-medium text-gray-900">
                                {file.source_file.replace('vietnamese_tiki_products_', '').replace('.csv', '')}
                              </span>
                            </div>
                            <span className="text-sm text-gray-600">
                              {file.count.toLocaleString()} records
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-12">
                  <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Failed to load database statistics</p>
                </div>
              )}
            </div>
          )}

          {/* SQL Query Tab */}
          {activeTab === 'query' && (
            <div className="space-y-6">
              {/* Predefined Queries */}
              <div className="bg-white rounded-lg shadow">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-medium text-gray-900">Quick Queries</h3>
                </div>
                <div className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {predefinedQueries.map((query, idx) => (
                      <button
                        key={idx}
                        onClick={() => setSqlQuery(query.query)}
                        className="text-left p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors"
                      >
                        <div className="font-medium text-gray-900">{query.name}</div>
                        <div className="text-sm text-gray-600 mt-1 font-mono truncate">
                          {query.query}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Query Editor */}
              <div className="bg-white rounded-lg shadow">
                <div className="px-6 py-4 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-medium text-gray-900">SQL Query Editor</h3>
                    <button
                      onClick={executeQuery}
                      disabled={queryLoading || !sqlQuery.trim()}
                      className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                    >
                      <Play className={`w-4 h-4 mr-2 ${queryLoading ? 'animate-spin' : ''}`} />
                      Execute
                    </button>
                  </div>
                </div>
                <div className="p-6">
                  <textarea
                    value={sqlQuery}
                    onChange={(e) => setSqlQuery(e.target.value)}
                    className="w-full h-32 p-4 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter your SQL query here..."
                  />
                </div>
              </div>

              {/* Query Results */}
              {queryResult && (
                <div className="bg-white rounded-lg shadow">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-medium text-gray-900">Query Results</h3>
                      {queryResult.results && (
                        <button className="flex items-center px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200">
                          <Download className="w-4 h-4 mr-1" />
                          Export
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="p-6">
                    {queryResult.error ? (
                      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div className="text-red-800 font-medium">Query Error</div>
                        <div className="text-red-700 text-sm mt-1">{queryResult.error}</div>
                      </div>
                    ) : queryResult.results ? (
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              {Object.keys(queryResult.results[0] || {}).map((column) => (
                                <th
                                  key={column}
                                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                                >
                                  {column}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {queryResult.results.slice(0, 100).map((row, idx) => (
                              <tr key={idx} className="hover:bg-gray-50">
                                {Object.values(row).map((value, colIdx) => (
                                  <td key={colIdx} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    {value?.toString() || ''}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {queryResult.results.length > 100 && (
                          <div className="text-center py-4 text-sm text-gray-500">
                            Showing first 100 rows of {queryResult.results.length} results
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <Eye className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p>Execute a query to see results</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Data Browser Tab */}
          {activeTab === 'data' && (
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium text-gray-900">Sample Data</h3>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => fetchSampleData(Math.max(1, currentPage - 1))}
                      disabled={currentPage <= 1 || loading}
                      className="p-2 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="text-sm text-gray-600">
                      Page {currentPage}
                    </span>
                    <button
                      onClick={() => fetchSampleData(currentPage + 1)}
                      disabled={loading}
                      className="p-2 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
              <div className="overflow-x-auto">
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    <span className="ml-2 text-gray-600">Loading data...</span>
                  </div>
                ) : sampleData.length > 0 ? (
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Price</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Brand</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Category</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rating</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {sampleData.map((product, idx) => (
                        <tr key={idx} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {product.tiki_id || product.id}
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-900 max-w-xs truncate">
                            {product.name}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {product.price?.toLocaleString()} ₫
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {product.brand}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {product.category}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {product.rating_average || 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="text-center py-12">
                    <Table className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">No data available</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Schema Tab */}
          {activeTab === 'schema' && (
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">Normalized Database Schema</h3>
                <p className="text-sm text-gray-600 mt-1">6-table normalized structure for complex JOIN queries</p>
              </div>
              <div className="p-6">
                <div className="space-y-6">
                  
                  {/* Brands Table */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-blue-50 px-4 py-3 border-b border-gray-200">
                      <h4 className="font-medium text-gray-900">brands</h4>
                      <p className="text-sm text-gray-600">Brand reference table (824 records)</p>
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          { name: 'brand_id', type: 'INTEGER', desc: 'Primary key' },
                          { name: 'brand_name', type: 'TEXT', desc: 'Brand name (Nike, Adidas, etc.)' },
                          { name: 'product_count', type: 'INTEGER', desc: 'Number of products' }
                        ].map((column, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded">
                            <div className="font-mono text-sm font-medium text-gray-900">{column.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{column.type}</div>
                            <div className="text-xs text-gray-600 mt-1">{column.desc}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Categories Table */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-green-50 px-4 py-3 border-b border-gray-200">
                      <h4 className="font-medium text-gray-900">categories</h4>
                      <p className="text-sm text-gray-600">Product category reference table (155 records)</p>
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          { name: 'category_id', type: 'INTEGER', desc: 'Primary key' },
                          { name: 'category_name', type: 'TEXT', desc: 'Category name (Balo nữ, Giày dép nam, etc.)' },
                          { name: 'product_count', type: 'INTEGER', desc: 'Number of products' }
                        ].map((column, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded">
                            <div className="font-mono text-sm font-medium text-gray-900">{column.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{column.type}</div>
                            <div className="text-xs text-gray-600 mt-1">{column.desc}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Sellers Table */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-purple-50 px-4 py-3 border-b border-gray-200">
                      <h4 className="font-medium text-gray-900">sellers</h4>
                      <p className="text-sm text-gray-600">Seller information and statistics (3,807 records)</p>
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          { name: 'seller_id', type: 'INTEGER', desc: 'Primary key' },
                          { name: 'seller_name', type: 'TEXT', desc: 'Seller name' },
                          { name: 'product_count', type: 'INTEGER', desc: 'Number of products sold' },
                          { name: 'total_quantity_sold', type: 'INTEGER', desc: 'Total units sold' },
                          { name: 'avg_rating', type: 'REAL', desc: 'Average product rating' }
                        ].map((column, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded">
                            <div className="font-mono text-sm font-medium text-gray-900">{column.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{column.type}</div>
                            <div className="text-xs text-gray-600 mt-1">{column.desc}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Products Table */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-yellow-50 px-4 py-3 border-b border-gray-200">
                      <h4 className="font-medium text-gray-900">products</h4>
                      <p className="text-sm text-gray-600">Core product information with foreign keys (41,576 records)</p>
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          { name: 'product_id', type: 'INTEGER', desc: 'Primary key' },
                          { name: 'tiki_id', type: 'INTEGER', desc: 'Original Tiki product ID' },
                          { name: 'name', type: 'TEXT', desc: 'Product name' },
                          { name: 'description', type: 'TEXT', desc: 'Product description' },
                          { name: 'brand_id', type: 'INTEGER', desc: 'Foreign key → brands.brand_id' },
                          { name: 'category_id', type: 'INTEGER', desc: 'Foreign key → categories.category_id' },
                          { name: 'seller_id', type: 'INTEGER', desc: 'Foreign key → sellers.seller_id' },
                          { name: 'date_created', type: 'INTEGER', desc: 'Creation timestamp' },
                          { name: 'number_of_images', type: 'INTEGER', desc: 'Number of product images' }
                        ].map((column, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded">
                            <div className="font-mono text-sm font-medium text-gray-900">{column.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{column.type}</div>
                            <div className="text-xs text-gray-600 mt-1">{column.desc}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Product Pricing Table */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-red-50 px-4 py-3 border-b border-gray-200">
                      <h4 className="font-medium text-gray-900">product_pricing</h4>
                      <p className="text-sm text-gray-600">Price and sales information (83,206 records)</p>
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          { name: 'pricing_id', type: 'INTEGER', desc: 'Primary key' },
                          { name: 'product_id', type: 'INTEGER', desc: 'Foreign key → products.product_id' },
                          { name: 'price', type: 'REAL', desc: 'Current price in VND' },
                          { name: 'original_price', type: 'REAL', desc: 'Original price in VND' },
                          { name: 'discount_rate', type: 'REAL', desc: 'Discount percentage' },
                          { name: 'quantity_sold', type: 'INTEGER', desc: 'Units sold' },
                          { name: 'favourite_count', type: 'INTEGER', desc: 'Number of favorites' },
                          { name: 'pay_later', type: 'BOOLEAN', desc: 'Pay later option available' },
                          { name: 'vnd_cashback', type: 'INTEGER', desc: 'Cashback amount in VND' }
                        ].map((column, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded">
                            <div className="font-mono text-sm font-medium text-gray-900">{column.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{column.type}</div>
                            <div className="text-xs text-gray-600 mt-1">{column.desc}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Product Reviews Table */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-indigo-50 px-4 py-3 border-b border-gray-200">
                      <h4 className="font-medium text-gray-900">product_reviews</h4>
                      <p className="text-sm text-gray-600">Review and rating information (83,206 records)</p>
                    </div>
                    <div className="p-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          { name: 'review_id', type: 'INTEGER', desc: 'Primary key' },
                          { name: 'product_id', type: 'INTEGER', desc: 'Foreign key → products.product_id' },
                          { name: 'rating_average', type: 'REAL', desc: 'Average rating (0-5)' },
                          { name: 'review_count', type: 'INTEGER', desc: 'Number of reviews' },
                          { name: 'has_video', type: 'BOOLEAN', desc: 'Has video reviews' }
                        ].map((column, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded">
                            <div className="font-mono text-sm font-medium text-gray-900">{column.name}</div>
                            <div className="text-xs text-blue-600 mt-1">{column.type}</div>
                            <div className="text-xs text-gray-600 mt-1">{column.desc}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Schema Relationships */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-gray-100 px-4 py-3 border-b border-gray-200">
                      <h4 className="font-medium text-gray-900">Table Relationships</h4>
                      <p className="text-sm text-gray-600">Foreign key relationships for JOIN queries</p>
                    </div>
                    <div className="p-4">
                      <div className="space-y-3 text-sm">
                        <div className="flex items-center">
                          <span className="font-mono bg-yellow-100 px-2 py-1 rounded">products.brand_id</span>
                          <span className="mx-2">→</span>
                          <span className="font-mono bg-blue-100 px-2 py-1 rounded">brands.brand_id</span>
                        </div>
                        <div className="flex items-center">
                          <span className="font-mono bg-yellow-100 px-2 py-1 rounded">products.category_id</span>
                          <span className="mx-2">→</span>
                          <span className="font-mono bg-green-100 px-2 py-1 rounded">categories.category_id</span>
                        </div>
                        <div className="flex items-center">
                          <span className="font-mono bg-yellow-100 px-2 py-1 rounded">products.seller_id</span>
                          <span className="mx-2">→</span>
                          <span className="font-mono bg-purple-100 px-2 py-1 rounded">sellers.seller_id</span>
                        </div>
                        <div className="flex items-center">
                          <span className="font-mono bg-red-100 px-2 py-1 rounded">product_pricing.product_id</span>
                          <span className="mx-2">→</span>
                          <span className="font-mono bg-yellow-100 px-2 py-1 rounded">products.product_id</span>
                        </div>
                        <div className="flex items-center">
                          <span className="font-mono bg-indigo-100 px-2 py-1 rounded">product_reviews.product_id</span>
                          <span className="mx-2">→</span>
                          <span className="font-mono bg-yellow-100 px-2 py-1 rounded">products.product_id</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Database Schema Diagram */}
                  <div className="border border-gray-200 rounded-lg">
                    <div className="bg-gradient-to-r from-blue-50 to-purple-50 px-4 py-3 border-b border-gray-200">
                      <div className="flex items-center">
                        <GitBranch className="w-5 h-5 text-blue-600 mr-2" />
                        <h4 className="font-medium text-gray-900">Entity Relationship Diagram</h4>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">Visual representation of normalized database schema</p>
                    </div>
                    <div className="p-6">
                      <div className="bg-white border border-gray-100 rounded-lg p-4">
                        <div className="flex justify-center">
                          <img 
                            src="/images/tiki_database_schema.png" 
                            alt="Tiki Database Schema ERD"
                            className="max-w-full h-auto rounded-lg shadow-sm"
                            style={{ maxHeight: '600px' }}
                          />
                        </div>
                      </div>
                      <div className="mt-4 text-sm text-gray-600">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <h5 className="font-medium text-gray-900 mb-2">Key Features:</h5>
                            <ul className="space-y-1">
                              <li>• 6-table normalized structure</li>
                              <li>• Foreign key relationships</li>
                              <li>• Complex JOIN query support</li>
                            </ul>
                          </div>
                          <div>
                            <h5 className="font-medium text-gray-900 mb-2">Record Counts:</h5>
                            <ul className="space-y-1">
                              <li>• 824 brands, 155 categories</li>
                              <li>• 3,807 sellers, 41,576 products</li>
                              <li>• 83,206 pricing & review records</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  )
}
