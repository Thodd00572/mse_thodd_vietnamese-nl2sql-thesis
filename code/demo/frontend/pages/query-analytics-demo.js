import { useState, useEffect } from 'react';
import Head from 'next/head';
import Layout from '../components/Layout';
import styles from '../styles/QueryAnalyticsDemo.module.css';

export default function QueryAnalyticsDemo() {
  const [queryResults, setQueryResults] = useState(null);
  const [complexityReport, setComplexityReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedComplexity, setSelectedComplexity] = useState('all');
  const [selectedQuery, setSelectedQuery] = useState(null);
  const [executionLogs, setExecutionLogs] = useState([]);
  const [showLogs, setShowLogs] = useState(true);

  // Organized queries by complexity levels for analytics demonstration
  const queriesByComplexity = {
    'simple': [
      { vietnamese: 'Tìm áo thun', english: 'Find T-shirt', complexity: 'simple' },
      { vietnamese: 'Hiển thị giày', english: 'Show shoes', complexity: 'simple' },
      { vietnamese: 'Cho tôi xem túi xách', english: 'Show me handbags', complexity: 'simple' },
      { vietnamese: 'Tìm điện thoại', english: 'Find phones', complexity: 'simple' },
      { vietnamese: 'Xem laptop', english: 'View laptops', complexity: 'simple' },
      { vietnamese: 'Hiển thị đồng hồ', english: 'Show watches', complexity: 'simple' }
    ],
    'medium': [
      { vietnamese: 'Áo thun giá dưới 500k', english: 'T-shirts under 500k', complexity: 'medium' },
      { vietnamese: 'Giày dưới 2 triệu', english: 'Shoes under 2 million', complexity: 'medium' },
      { vietnamese: 'Điện thoại Samsung', english: 'Samsung phones', complexity: 'medium' },
      { vietnamese: 'Túi xách giá rẻ', english: 'Cheap handbags', complexity: 'medium' },
      { vietnamese: 'Giày Nike màu đen', english: 'Black Nike shoes', complexity: 'medium' },
      { vietnamese: 'Laptop có đánh giá cao', english: 'Highly rated laptops', complexity: 'medium' }
    ],
    'complex': [
      { vietnamese: 'Top 10 điện thoại có đánh giá cao nhất', english: 'Top 10 highest rated phones', complexity: 'complex' },
      { vietnamese: 'Áo thun nam màu xanh giá dưới 300k', english: 'Blue men t-shirts under 300k', complexity: 'complex' },
      { vietnamese: 'Sản phẩm Samsung hoặc Apple có đánh giá trên 4 sao', english: 'Samsung or Apple products with rating above 4 stars', complexity: 'complex' },
      { vietnamese: 'Giày thể thao nam giá từ 1 triệu đến 3 triệu có nhiều đánh giá', english: 'Men sports shoes priced 1-3 million with many reviews', complexity: 'complex' },
      { vietnamese: 'Top 5 túi xách nữ đắt nhất của thương hiệu Louis Vuitton', english: 'Top 5 most expensive Louis Vuitton women bags', complexity: 'complex' },
      { vietnamese: 'Điện thoại có camera tốt nhất dưới 15 triệu', english: 'Best camera phones under 15 million', complexity: 'complex' }
    ]
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = { timestamp, message, type, id: Date.now() };
    setExecutionLogs(prev => [newLog, ...prev.slice(0, 49)]); // Keep last 50 logs
  };

  const executeQuery = async (vietnameseQuery, pipeline = 'both') => {
    addLog(`🚀 Executing query: "${vietnameseQuery}" on ${pipeline === 'both' ? 'both pipelines' : `Pipeline ${pipeline}`}`, 'info');
    
    try {
      setLoading(true);
      
      if (pipeline === 'both' || pipeline === '1') {
        addLog('📡 Sending to Pipeline 1...', 'info');
        const response1 = await fetch('https://abnormally-direct-rhino.ngrok-free.app/pipeline1', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: vietnameseQuery })
        });
        const result1 = await response1.json();
        addLog(`📝 Pipeline 1 SQL: ${result1.sql_query}`, 'sql');
        addLog(`⏱️ Pipeline 1 Time: ${formatTime(result1.execution_time * 1000)}`, 'timing');
      }

      if (pipeline === 'both' || pipeline === '2') {
        addLog('📡 Sending to Pipeline 2...', 'info');
        const response2 = await fetch('https://abnormally-direct-rhino.ngrok-free.app/pipeline2', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: vietnameseQuery })
        });
        const result2 = await response2.json();
        addLog(`🌐 Pipeline 2 Translation: ${result2.english_translation}`, 'translation');
        addLog(`📝 Pipeline 2 SQL: ${result2.sql_query}`, 'sql');
        addLog(`⏱️ Pipeline 2 Time: ${formatTime(result2.execution_time * 1000)}`, 'timing');
      }

      addLog(`✅ Query execution completed successfully`, 'success');
      
    } catch (error) {
      addLog(`❌ Query execution failed: ${error.message}`, 'error');
      console.error('Error executing query:', error);
    }
    setLoading(false);
  };

  const executeAllQueriesInComplexity = async (complexity) => {
    setLoading(true);
    addLog(`🔄 Starting execution of all ${complexity} queries for analytics...`, 'info');
    
    const queries = queriesByComplexity[complexity];
    let successCount = 0;
    let errorCount = 0;

    for (const query of queries) {
      try {
        addLog(`🚀 Processing: "${query.vietnamese}"`, 'info');
        await executeQuery(query.vietnamese, 'both');
        successCount++;
      } catch (error) {
        errorCount++;
        addLog(`❌ Failed: "${query.vietnamese}" - ${error.message}`, 'error');
      }
    }

    addLog(`📊 ${complexity.toUpperCase()} queries completed: ${successCount} success, ${errorCount} errors`, 'result');
    setLoading(false);
  };

  const executeAllQueries = async () => {
    setLoading(true);
    addLog('🔄 Starting comprehensive analytics demonstration - executing all queries...', 'info');
    
    try {
      const response = await fetch('https://abnormally-direct-rhino.ngrok-free.app/api/sample-queries/execute-samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      setQueryResults(data);
      
      addLog(`✅ Analytics demonstration completed: ${data.results?.length || 0} queries processed`, 'success');
      
      // Get complexity report for analytics
      const reportResponse = await fetch('https://abnormally-direct-rhino.ngrok-free.app/api/sample-queries/complexity-report');
      const reportData = await reportResponse.json();
      setComplexityReport(reportData);
      
      addLog('📊 Analytics report generated successfully', 'info');
      
    } catch (error) {
      addLog(`❌ Analytics demonstration failed: ${error.message}`, 'error');
      console.error('Error executing queries:', error);
    }
    setLoading(false);
  };

  const clearLogs = () => {
    setExecutionLogs([]);
    addLog('🧹 Analytics logs cleared', 'info');
  };

  const getFilteredQueries = () => {
    if (selectedComplexity === 'all') {
      return Object.entries(queriesByComplexity);
    }
    return [[selectedComplexity, queriesByComplexity[selectedComplexity]]];
  };

  const complexityOptions = [
    { value: 'all', label: 'All Complexity Levels', count: Object.values(queriesByComplexity).flat().length },
    { value: 'simple', label: 'Simple Queries', count: queriesByComplexity.simple.length },
    { value: 'medium', label: 'Medium Queries', count: queriesByComplexity.medium.length },
    { value: 'complex', label: 'Complex Queries', count: queriesByComplexity.complex.length }
  ];

  const getComplexityColor = (complexity) => {
    switch (complexity) {
      case 'simple': return '#4CAF50';
      case 'medium': return '#FF9800';
      case 'complex': return '#F44336';
      default: return '#2196F3';
    }
  };

  const formatTime = (timeMs) => {
    return timeMs < 1000 ? `${timeMs.toFixed(1)}ms` : `${(timeMs/1000).toFixed(2)}s`;
  };

  const getLogIcon = (type) => {
    switch (type) {
      case 'success': return '✅';
      case 'error': return '❌';
      case 'info': return 'ℹ️';
      case 'sql': return '📝';
      case 'translation': return '🌐';
      case 'result': return '📊';
      case 'timing': return '⏱️';
      default: return '📋';
    }
  };

  return (
    <Layout>
      <Head>
        <title>Query Analytics Demonstration - Vietnamese NL2SQL System</title>
        <meta name="description" content="Comprehensive analytics demonstration of Vietnamese NL2SQL query execution across complexity levels" />
      </Head>

      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.titleSection}>
            <h1>Query Analytics Demonstration</h1>
            <p>Comprehensive execution and analysis of Vietnamese NL2SQL queries across complexity levels for performance analytics and system demonstration</p>
            <div className={styles.statsBar}>
              <div className={styles.stat}>
                <span className={styles.statNumber}>{Object.values(queriesByComplexity).flat().length}</span>
                <span className={styles.statLabel}>Total Queries</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>3</span>
                <span className={styles.statLabel}>Complexity Levels</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>2</span>
                <span className={styles.statLabel}>Pipeline Systems</span>
              </div>
            </div>
          </div>
        </div>

        <div className={styles.mainContent}>
          <div className={styles.leftPanel}>
            <div className={styles.controlSection}>
              <h2>Analytics Controls</h2>
              
              {/* Complexity Filter */}
              <div className={styles.filterGroup}>
                <label htmlFor="complexity-select">Filter by Complexity Level:</label>
                <select
                  id="complexity-select"
                  value={selectedComplexity}
                  onChange={(e) => setSelectedComplexity(e.target.value)}
                  className={styles.dropdown}
                >
                  {complexityOptions.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label} ({option.count} queries)
                    </option>
                  ))}
                </select>
              </div>

              {/* Execution Controls */}
              <div className={styles.executionControls}>
                <button
                  onClick={executeAllQueries}
                  disabled={loading}
                  className={`${styles.executeButton} ${styles.primary}`}
                >
                  {loading ? '🔄 Running Analytics...' : '🚀 Execute All Queries for Analytics'}
                </button>
                
                {selectedComplexity !== 'all' && (
                  <button
                    onClick={() => executeAllQueriesInComplexity(selectedComplexity)}
                    disabled={loading}
                    className={`${styles.executeButton} ${styles.secondary}`}
                  >
                    {loading ? '🔄 Processing...' : `Execute All ${selectedComplexity.toUpperCase()} Queries`}
                  </button>
                )}
              </div>
            </div>

            {/* Query Sections */}
            <div className={styles.querySections}>
              <h2>Query Collections</h2>
              {getFilteredQueries().map(([complexity, queries]) => (
                <div key={complexity} className={styles.complexitySection}>
                  <div 
                    className={styles.complexityHeader}
                    style={{ borderLeftColor: getComplexityColor(complexity) }}
                  >
                    <h3>{complexity.toUpperCase()} QUERIES</h3>
                    <span className={styles.queryCount}>{queries.length} queries</span>
                  </div>
                  
                  <div className={styles.queryList}>
                    {queries.map((query, index) => (
                      <div 
                        key={index} 
                        className={styles.queryItem}
                        onClick={() => setSelectedQuery(query)}
                      >
                        <div className={styles.queryContent}>
                          <div className={styles.vietnameseQuery}>{query.vietnamese}</div>
                          <div className={styles.englishQuery}>{query.english}</div>
                        </div>
                        <div className={styles.queryActions}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              executeQuery(query.vietnamese, '1');
                            }}
                            disabled={loading}
                            className={styles.pipelineButton}
                          >
                            P1
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              executeQuery(query.vietnamese, '2');
                            }}
                            disabled={loading}
                            className={styles.pipelineButton}
                          >
                            P2
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              executeQuery(query.vietnamese, 'both');
                            }}
                            disabled={loading}
                            className={styles.pipelineButton}
                          >
                            Both
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.rightPanel}>
            {/* Execution Logs */}
            <div className={styles.logsSection}>
              <div className={styles.logsHeader}>
                <h2>Analytics Execution Logs</h2>
                <div className={styles.logsControls}>
                  <button onClick={() => setShowLogs(!showLogs)} className={styles.toggleButton}>
                    {showLogs ? '🙈 Hide Logs' : '👁️ Show Logs'}
                  </button>
                  <button onClick={clearLogs} className={styles.clearButton}>
                    🧹 Clear Logs
                  </button>
                </div>
              </div>
              
              {showLogs && (
                <div className={styles.logsContainer}>
                  {executionLogs.length === 0 ? (
                    <div className={styles.emptyLogs}>
                      <p>📋 No execution logs yet</p>
                      <p>Execute queries to see detailed analytics logs here</p>
                    </div>
                  ) : (
                    <div className={styles.logsList}>
                      {executionLogs.map(log => (
                        <div key={log.id} className={`${styles.logEntry} ${styles[log.type]}`}>
                          <span className={styles.logIcon}>{getLogIcon(log.type)}</span>
                          <span className={styles.logTimestamp}>{log.timestamp}</span>
                          <span className={styles.logMessage}>{log.message}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Analytics Results */}
            {queryResults && (
              <div className={styles.resultsSection}>
                <h2>Analytics Results</h2>
                <div className={styles.analyticsOverview}>
                  <div className={styles.analyticsCard}>
                    <h3>Execution Summary</h3>
                    <p>Total Queries: {queryResults.results?.length || 0}</p>
                    <p>Success Rate: {queryResults.success_rate || 0}%</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
