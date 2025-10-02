import { useState, useEffect } from 'react';
import Head from 'next/head';
import Layout from '../components/Layout';
import styles from '../styles/SampleQueries.module.css';

export default function SampleQueries() {
  const [queryResults, setQueryResults] = useState(null);
  const [complexityReport, setComplexityReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedComplexity, setSelectedComplexity] = useState('all');
  const [selectedQuery, setSelectedQuery] = useState(null);
  const [executionLogs, setExecutionLogs] = useState([]);
  const [showLogs, setShowLogs] = useState(true);
  const [sampleQueries, setSampleQueries] = useState([]);

  // Load sample queries data from backend - 100 queries each level
  const [sampleQueriesData, setSampleQueriesData] = useState({
    'simple': [],
    'medium': [],
    'complex': []
  });

  // Load sample queries on component mount
  useEffect(() => {
    const loadSampleQueries = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/sample-queries');
        if (response.ok) {
          const data = await response.json();
          setSampleQueriesData(data);
        } else {
          // Fallback to local data if API fails
          setSampleQueriesData(getLocalSampleQueries());
        }
      } catch (error) {
        console.error('Failed to load sample queries:', error);
        setSampleQueriesData(getLocalSampleQueries());
      }
    };
    
    loadSampleQueries();
  }, []);

  // Fallback local sample queries data
  const getLocalSampleQueries = () => ({
    'simple': [
      { vietnamese: 'Tìm áo thun', english: 'Find T-shirt', sql: "SELECT * FROM products WHERE name LIKE '%áo thun%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Hiển thị giày', english: 'Show shoes', sql: "SELECT * FROM products WHERE name LIKE '%giày%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Cho tôi xem túi xách', english: 'Show me handbags', sql: "SELECT * FROM products WHERE name LIKE '%túi%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Tìm điện thoại', english: 'Find phones', sql: "SELECT * FROM products WHERE name LIKE '%điện thoại%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Xem balo', english: 'View backpacks', sql: "SELECT * FROM products WHERE name LIKE '%balo%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Hiển thị vali', english: 'Show suitcases', sql: "SELECT * FROM products WHERE name LIKE '%vali%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Tìm ví', english: 'Find wallets', sql: "SELECT * FROM products WHERE name LIKE '%ví%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Xem dép', english: 'View sandals', sql: "SELECT * FROM products WHERE name LIKE '%dép%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Hiển thị nón', english: 'Show hats', sql: "SELECT * FROM products WHERE name LIKE '%nón%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Tìm thắt lưng', english: 'Find belts', sql: "SELECT * FROM products WHERE name LIKE '%thắt lưng%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Xem kính', english: 'View glasses', sql: "SELECT * FROM products WHERE name LIKE '%kính%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Hiển thị đồng hồ', english: 'Show watches', sql: "SELECT * FROM products WHERE name LIKE '%đồng hồ%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Tìm vớ', english: 'Find socks', sql: "SELECT * FROM products WHERE name LIKE '%vớ%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Xem khăn', english: 'View scarves', sql: "SELECT * FROM products WHERE name LIKE '%khăn%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Hiển thị găng tay', english: 'Show gloves', sql: "SELECT * FROM products WHERE name LIKE '%găng tay%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Tìm áo khoác', english: 'Find jackets', sql: "SELECT * FROM products WHERE name LIKE '%áo khoác%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Xem quần', english: 'View pants', sql: "SELECT * FROM products WHERE name LIKE '%quần%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Hiển thị váy', english: 'Show dresses', sql: "SELECT * FROM products WHERE name LIKE '%váy%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Tìm áo sơ mi', english: 'Find shirts', sql: "SELECT * FROM products WHERE name LIKE '%áo sơ mi%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Xem giày thể thao', english: 'View sneakers', sql: "SELECT * FROM products WHERE name LIKE '%giày thể thao%' LIMIT 10;", complexity: 'simple' },
      { vietnamese: 'Tìm mũ lưỡi trai', english: 'Find caps', complexity: 'simple' },
      { vietnamese: 'Xem mũ beret', english: 'View berets', complexity: 'simple' },
      { vietnamese: 'Hiển thị áo len', english: 'Show sweaters', complexity: 'simple' },
      { vietnamese: 'Tìm áo hoodie', english: 'Find hoodies', complexity: 'simple' },
      { vietnamese: 'Xem quần jean', english: 'View jeans', complexity: 'simple' },
      { vietnamese: 'Hiển thị quần short', english: 'Show shorts', complexity: 'simple' },
      { vietnamese: 'Tìm áo polo', english: 'Find polo shirts', complexity: 'simple' },
      { vietnamese: 'Xem áo tank top', english: 'View tank tops', complexity: 'simple' },
      { vietnamese: 'Hiển thị đầm', english: 'Show gowns', complexity: 'simple' },
      { vietnamese: 'Tìm chân váy', english: 'Find skirts', sql: "SELECT * FROM products WHERE name LIKE '%chân váy%' LIMIT 10;", complexity: 'simple' }
    ],
    'medium': [
      { vietnamese: 'Áo thun giá dưới 500k', english: 'T-shirts under 500k', sql: 'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE \'%áo thun%\' AND pr.current_price < 500000 ORDER BY pr.current_price LIMIT 20;', complexity: 'medium' },
      { vietnamese: 'Giày dưới 2 triệu', english: 'Shoes under 2 million', sql: 'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE \'%giày%\' AND pr.current_price < 2000000 ORDER BY pr.current_price LIMIT 20;', complexity: 'medium' },
      { vietnamese: 'Túi xách giá rẻ', english: 'Cheap handbags', sql: 'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE \'%túi%\' ORDER BY pr.current_price LIMIT 20;', complexity: 'medium' },
      { vietnamese: 'Điện thoại Apple', english: 'Apple phones', sql: 'SELECT p.name, b.brand_name, pr.current_price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE \'%điện thoại%\' AND b.brand_name LIKE \'%Apple%\' LIMIT 10;', complexity: 'medium' },
      { vietnamese: 'Giày Nike', english: 'Nike shoes', sql: 'SELECT p.name, b.brand_name, pr.current_price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE \'%giày%\' AND b.brand_name LIKE \'%Nike%\' LIMIT 10;', complexity: 'medium' },
      { vietnamese: 'Túi Louis Vuitton', english: 'Louis Vuitton bags', complexity: 'medium' },
      { vietnamese: 'Áo thun màu đen', english: 'Black t-shirts', complexity: 'medium' },
      { vietnamese: 'Giày size 42', english: 'Size 42 shoes', complexity: 'medium' },
      { vietnamese: 'Túi xách màu nâu', english: 'Brown handbags', complexity: 'medium' },
      { vietnamese: 'Điện thoại Samsung Galaxy', english: 'Samsung Galaxy phones', complexity: 'medium' },
      { vietnamese: 'Giày Adidas trắng', english: 'White Adidas shoes', complexity: 'medium' },
      { vietnamese: 'Balo đi học', english: 'School backpacks', complexity: 'medium' },
      { vietnamese: 'Vali kéo', english: 'Rolling suitcases', complexity: 'medium' },
      { vietnamese: 'Ví nam da', english: 'Leather men wallets', complexity: 'medium' },
      { vietnamese: 'Dép quai ngang', english: 'Slide sandals', complexity: 'medium' },
      { vietnamese: 'Nón snapback', english: 'Snapback caps', complexity: 'medium' },
      { vietnamese: 'Thắt lưng da', english: 'Leather belts', complexity: 'medium' },
      { vietnamese: 'Kính râm', english: 'Sunglasses', complexity: 'medium' },
      { vietnamese: 'Đồng hồ thông minh', english: 'Smart watches', complexity: 'medium' },
      { vietnamese: 'Vớ cotton', english: 'Cotton socks', complexity: 'medium' },
      { vietnamese: 'Khăn lụa', english: 'Silk scarves', complexity: 'medium' },
      { vietnamese: 'Áo khoác jean', english: 'Denim jackets', complexity: 'medium' },
      { vietnamese: 'Quần jogger', english: 'Jogger pants', complexity: 'medium' },
      { vietnamese: 'Váy maxi', english: 'Maxi dresses', complexity: 'medium' },
      { vietnamese: 'Áo sơ mi trắng', english: 'White shirts', complexity: 'medium' },
      { vietnamese: 'Giày thể thao nam', english: 'Men sneakers', complexity: 'medium' },
      { vietnamese: 'Giày cao gót nữ', english: 'Women high heels', complexity: 'medium' },
      { vietnamese: 'Boot da', english: 'Leather boots', complexity: 'medium' },
      { vietnamese: 'Túi đeo chéo nữ', english: 'Women crossbody bags', complexity: 'medium' },
      { vietnamese: 'Cặp laptop 15 inch', english: '15 inch laptop bags', complexity: 'medium' },
      { vietnamese: 'Phụ kiện tóc', english: 'Hair accessories', complexity: 'medium' },
      { vietnamese: 'Trang sức bạc', english: 'Silver jewelry', complexity: 'medium' },
      { vietnamese: 'Mũ len', english: 'Knit hats', complexity: 'medium' },
      { vietnamese: 'Áo len cổ lọ', english: 'Turtleneck sweaters', complexity: 'medium' },
      { vietnamese: 'Hoodie có mũ', english: 'Hooded sweatshirts', complexity: 'medium' },
      { vietnamese: 'Jean skinny', english: 'Skinny jeans', complexity: 'medium' },
      { vietnamese: 'Quần short jean', english: 'Denim shorts', complexity: 'medium' },
      { vietnamese: 'Áo polo nam', english: 'Men polo shirts', complexity: 'medium' },
      { vietnamese: 'Tank top nữ', english: 'Women tank tops', complexity: 'medium' },
      { vietnamese: 'Đầm công sở', english: 'Office dresses', complexity: 'medium' },
      { vietnamese: 'Chân váy bút chì', english: 'Pencil skirts', complexity: 'medium' },
      { vietnamese: 'Blazer nữ', english: 'Women blazers', complexity: 'medium' },
      { vietnamese: 'Áo vest nam', english: 'Men vests', complexity: 'medium' },
      { vietnamese: 'Đồ lót cotton', english: 'Cotton underwear', complexity: 'medium' },
      { vietnamese: 'Đồ ngủ lụa', english: 'Silk sleepwear', complexity: 'medium' },
      { vietnamese: 'Bikini hai mảnh', english: 'Two piece bikinis', complexity: 'medium' },
      { vietnamese: 'Đồ thể thao nam', english: 'Men sportswear', complexity: 'medium' },
      { vietnamese: 'Quần yoga nữ', english: 'Women yoga pants', complexity: 'medium' },
      { vietnamese: 'Áo gym dri-fit', english: 'Dri-fit gym shirts', complexity: 'medium' },
      { vietnamese: 'Giày chạy bộ Nike', english: 'Nike running shoes', complexity: 'medium' }
    ],
    'complex': [
      { vietnamese: 'Top 10 điện thoại có đánh giá cao nhất', english: 'Top 10 highest rated phones', complexity: 'complex' },
      { vietnamese: 'Áo thun nam màu xanh giá dưới 300k', english: 'Blue men t-shirts under 300k', complexity: 'complex' },
      { vietnamese: 'Sản phẩm Samsung hoặc Apple có đánh giá trên 4 sao', english: 'Samsung or Apple products with rating above 4 stars', complexity: 'complex' },
      { vietnamese: 'Giày Nike hoặc Adidas giá từ 1 triệu đến 3 triệu', english: 'Nike or Adidas shoes priced 1-3 million', complexity: 'complex' },
      { vietnamese: 'Top 5 túi xách nữ bán chạy nhất', english: 'Top 5 best selling women handbags', complexity: 'complex' },
      { vietnamese: 'Áo khoác nam size L màu đen giá dưới 800k', english: 'Black men jackets size L under 800k', complexity: 'complex' },
      { vietnamese: 'Sản phẩm có đánh giá trên 4.5 sao và bán trên 100 sản phẩm', english: 'Products with rating above 4.5 stars and sales over 100', complexity: 'complex' },
      { vietnamese: 'Điện thoại iPhone hoặc Samsung giá dưới 15 triệu', english: 'iPhone or Samsung phones under 15 million', complexity: 'complex' },
      { vietnamese: 'Top 10 sản phẩm có giá cao nhất', english: 'Top 10 highest priced products', complexity: 'complex' },
      { vietnamese: 'Giày thể thao nam size 41-43 giá từ 500k đến 2 triệu', english: 'Men sneakers size 41-43 priced 500k-2 million', complexity: 'complex' },
      { vietnamese: 'Túi xách Louis Vuitton hoặc Gucci có đánh giá trên 4 sao', english: 'Louis Vuitton or Gucci bags with rating above 4 stars', complexity: 'complex' },
      { vietnamese: 'Sản phẩm bán chạy nhất trong tháng', english: 'Best selling products this month', complexity: 'complex' },
      { vietnamese: 'Áo thun nữ màu trắng hoặc hồng size S giá dưới 200k', english: 'White or pink women t-shirts size S under 200k', complexity: 'complex' },
      { vietnamese: 'Top 5 thương hiệu có nhiều sản phẩm nhất', english: 'Top 5 brands with most products', complexity: 'complex' },
      { vietnamese: 'Giày cao gót nữ màu đen size 36-38 giá từ 300k đến 1 triệu', english: 'Black women high heels size 36-38 priced 300k-1 million', complexity: 'complex' },
      { vietnamese: 'Sản phẩm Nike hoặc Adidas có giá giảm trên 20%', english: 'Nike or Adidas products with discount over 20%', complexity: 'complex' },
      { vietnamese: 'Balo laptop 15 inch hoặc 17 inch giá dưới 1 triệu', english: '15 or 17 inch laptop backpacks under 1 million', complexity: 'complex' },
      { vietnamese: 'Top 10 sản phẩm có nhiều đánh giá nhất', english: 'Top 10 most reviewed products', complexity: 'complex' },
      { vietnamese: 'Vali kéo size 20-24 inch giá từ 800k đến 2 triệu', english: 'Rolling suitcases 20-24 inch priced 800k-2 million', complexity: 'complex' },
      { vietnamese: 'Ví nam da thật hoặc da PU giá dưới 500k', english: 'Real leather or PU men wallets under 500k', complexity: 'complex' },
      { vietnamese: 'Sản phẩm có đánh giá từ 4-5 sao và giá dưới 1 triệu', english: 'Products with 4-5 star rating and price under 1 million', complexity: 'complex' },
      { vietnamese: 'Dép quai ngang nam size 40-42 giá từ 100k đến 500k', english: 'Men slide sandals size 40-42 priced 100k-500k', complexity: 'complex' },
      { vietnamese: 'Top 5 danh mục sản phẩm có nhiều sản phẩm nhất', english: 'Top 5 categories with most products', complexity: 'complex' },
      { vietnamese: 'Nón snapback hoặc bucket hat màu đen giá dưới 300k', english: 'Black snapback or bucket hats under 300k', complexity: 'complex' },
      { vietnamese: 'Thắt lưng da nam màu nâu hoặc đen giá từ 200k đến 800k', english: 'Brown or black leather men belts priced 200k-800k', complexity: 'complex' },
      { vietnamese: 'Kính râm nam hoặc nữ có UV protection giá dưới 1 triệu', english: 'Men or women sunglasses with UV protection under 1 million', complexity: 'complex' },
      { vietnamese: 'Đồng hồ thông minh Apple hoặc Samsung giá dưới 10 triệu', english: 'Apple or Samsung smartwatches under 10 million', complexity: 'complex' },
      { vietnamese: 'Sản phẩm mới nhất trong 30 ngày có đánh giá trên 4 sao', english: 'Latest products in 30 days with rating above 4 stars', complexity: 'complex' },
      { vietnamese: 'Vớ cotton nam hoặc nữ pack 5-10 đôi giá dưới 200k', english: 'Cotton socks men or women 5-10 pack under 200k', complexity: 'complex' },
      { vietnamese: 'Khăn lụa nữ màu đỏ hoặc xanh giá từ 100k đến 500k', english: 'Red or blue women silk scarves priced 100k-500k', complexity: 'complex' },
      { vietnamese: 'Top 10 sản phẩm giảm giá nhiều nhất', english: 'Top 10 most discounted products', complexity: 'complex' },
      { vietnamese: 'Áo khoác jean nam hoặc nữ size M-XL giá dưới 600k', english: 'Men or women denim jackets size M-XL under 600k', complexity: 'complex' },
      { vietnamese: 'Quần jogger nam màu xám hoặc đen size L-XXL', english: 'Gray or black men jogger pants size L-XXL', complexity: 'complex' },
      { vietnamese: 'Váy maxi nữ màu hoa hoặc trơn giá từ 300k đến 1 triệu', english: 'Floral or solid women maxi dresses priced 300k-1 million', complexity: 'complex' },
      { vietnamese: 'Áo sơ mi trắng nam hoặc nữ size S-L giá dưới 400k', english: 'White shirts men or women size S-L under 400k', complexity: 'complex' },
      { vietnamese: 'Giày thể thao Nike Air Max hoặc Adidas Ultraboost', english: 'Nike Air Max or Adidas Ultraboost sneakers', complexity: 'complex' },
      { vietnamese: 'Sản phẩm có giá từ 500k đến 2 triệu và đánh giá trên 4.2 sao', english: 'Products priced 500k-2 million with rating above 4.2 stars', complexity: 'complex' },
      { vietnamese: 'Boot da nam hoặc nữ màu nâu size 38-42 giá dưới 1.5 triệu', english: 'Brown leather boots men or women size 38-42 under 1.5 million', complexity: 'complex' },
      { vietnamese: 'Túi đeo chéo nữ màu đen hoặc nâu có nhiều ngăn', english: 'Black or brown women crossbody bags with multiple compartments', complexity: 'complex' },
      { vietnamese: 'Top 5 sản phẩm có tỷ lệ đánh giá 5 sao cao nhất', english: 'Top 5 products with highest 5-star rating percentage', complexity: 'complex' },
      { vietnamese: 'Cặp laptop chống sốc 13-15 inch giá từ 400k đến 1 triệu', english: 'Shock-proof laptop bags 13-15 inch priced 400k-1 million', complexity: 'complex' },
      { vietnamese: 'Phụ kiện tóc nữ màu vàng hoặc bạc giá dưới 150k', english: 'Gold or silver women hair accessories under 150k', complexity: 'complex' },
      { vietnamese: 'Trang sức bạc 925 hoặc vàng 18k có đánh giá trên 4.5 sao', english: '925 silver or 18k gold jewelry with rating above 4.5 stars', complexity: 'complex' },
      { vietnamese: 'Mũ len nam hoặc nữ màu đen giá từ 50k đến 300k', english: 'Black knit hats men or women priced 50k-300k', complexity: 'complex' },
      { vietnamese: 'Áo len cổ lọ nữ màu be hoặc xám size S-M', english: 'Beige or gray women turtleneck sweaters size S-M', complexity: 'complex' },
      { vietnamese: 'Hoodie unisex có logo thương hiệu giá dưới 800k', english: 'Unisex hoodies with brand logo under 800k', complexity: 'complex' },
      { vietnamese: 'Jean skinny nữ màu xanh hoặc đen size 26-30', english: 'Blue or black women skinny jeans size 26-30', complexity: 'complex' },
      { vietnamese: 'Quần short jean nam size 30-34 giá từ 200k đến 600k', english: 'Men denim shorts size 30-34 priced 200k-600k', complexity: 'complex' },
      { vietnamese: 'Áo polo nam có cổ màu trắng hoặc xanh navy', english: 'White or navy blue men polo shirts with collar', complexity: 'complex' },
      { vietnamese: 'Tank top nữ cotton màu pastel giá dưới 200k', english: 'Pastel cotton women tank tops under 200k', complexity: 'complex' }
    ]
  });

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = { timestamp, message, type, id: Date.now() };
    setExecutionLogs(prev => [newLog, ...prev.slice(0, 49)]); // Keep last 50 logs
  };

  const executeQuery = async (vietnameseQuery, pipeline = 'both') => {
    addLog(`🚀 Executing query: "${vietnameseQuery}" on ${pipeline === 'both' ? 'both pipelines' : `Pipeline ${pipeline}`}`, 'info');
    
    try {
      if (pipeline === 'both' || pipeline === '1') {
        addLog('📡 API Request: POST /api/search (Pipeline 1)', 'api');
        addLog(`📤 Request Body: {"query": "${vietnameseQuery}", "pipeline": "pipeline1"}`, 'request');
        
        const p1Response = await fetch('http://localhost:8000/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            query: vietnameseQuery, 
            pipeline: 'pipeline1' 
          })
        });
        const p1Data = await p1Response.json();
        
        addLog(`📥 API Response: ${p1Response.status} ${p1Response.statusText}`, 'response');
        
        if (p1Data.success) {
          addLog(`✅ Pipeline 1 Success: Generated SQL in ${(p1Data.pipeline1_result.execution_time * 1000).toFixed(1)}ms`, 'success');
          addLog(`📝 Generated SQL: ${p1Data.pipeline1_result.sql_query}`, 'sql');
          addLog(`📊 Query Results: ${p1Data.pipeline1_result.results.length} rows returned`, 'result');
        } else {
          addLog(`❌ Pipeline 1 Error: ${p1Data.error}`, 'error');
        }
      }
      
      if (pipeline === 'both' || pipeline === '2') {
        addLog('📡 API Request: POST /api/search (Pipeline 2)', 'api');
        addLog(`📤 Request Body: {"query": "${vietnameseQuery}", "pipeline": "pipeline2"}`, 'request');
        
        const p2Response = await fetch('http://localhost:8000/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            query: vietnameseQuery, 
            pipeline: 'pipeline2' 
          })
        });
        const p2Data = await p2Response.json();
        
        addLog(`📥 API Response: ${p2Response.status} ${p2Response.statusText}`, 'response');
        
        if (p2Data.success) {
          addLog(`🌐 Translation: "${vietnameseQuery}" → "${p2Data.pipeline2_result.english_query}"`, 'translation');
          addLog(`✅ Pipeline 2 Success: Generated SQL in ${(p2Data.pipeline2_result.execution_time * 1000).toFixed(1)}ms`, 'success');
          addLog(`📝 Generated SQL: ${p2Data.pipeline2_result.sql_query}`, 'sql');
          addLog(`📊 Query Results: ${p2Data.pipeline2_result.results.length} rows returned`, 'result');
          addLog(`⏱️ Timing Breakdown: VN→EN ${(p2Data.pipeline2_result.vn_en_time * 1000).toFixed(1)}ms, EN→SQL ${(p2Data.pipeline2_result.en_sql_time * 1000).toFixed(1)}ms`, 'timing');
        } else {
          addLog(`❌ Pipeline 2 Error: ${p2Data.error}`, 'error');
        }
      }
      
      addLog(`🎉 Query execution completed successfully!`, 'success');
      
    } catch (error) {
      addLog(`💥 Execution failed: ${error.message}`, 'error');
    }
  };

  const executeAllQueries = async () => {
    setLoading(true);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `batch_execution_${timestamp}.json`;
    
    addLog('🔄 Starting sequential batch execution of all sample queries...', 'info');
    addLog(`📁 Results will be saved to: ${filename}`, 'info');
    
    const allQueries = getFilteredSampleQueries();
    const batchResults = {
      execution_timestamp: new Date().toISOString(),
      total_queries: allQueries.length,
      complexity_filter: selectedComplexity,
      results: [],
      summary: {
        pipeline1: { success: 0, failed: 0, total_time: 0 },
        pipeline2: { success: 0, failed: 0, total_time: 0 }
      }
    };
    
    try {
      for (let i = 0; i < allQueries.length; i++) {
        const query = allQueries[i];
        const queryId = `${query.complexity}_${i + 1}`;
        
        addLog(`🔄 [${i + 1}/${allQueries.length}] Processing: "${query.vietnamese}"`, 'info');
        
        const queryResult = {
          query_id: queryId,
          vietnamese_query: query.vietnamese,
          english_query: query.english,
          complexity: query.complexity,
          pipeline1_result: null,
          pipeline2_result: null,
          execution_time: new Date().toISOString()
        };
        
        // Execute Pipeline 1
        try {
          addLog(`📡 Pipeline 1 API Request: POST /api/search`, 'api');
          addLog(`📤 Request Body: {"query": "${query.vietnamese}", "pipeline": "pipeline1"}`, 'request');
          
          const p1Response = await fetch('http://localhost:8000/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              query: query.vietnamese, 
              pipeline: 'pipeline1' 
            })
          });
          const p1Data = await p1Response.json();
          
          addLog(`📥 Pipeline 1 Response: ${p1Response.status} ${p1Response.statusText}`, 'response');
          addLog(`📋 Response Data: ${JSON.stringify(p1Data, null, 2)}`, 'response');
          
          if (p1Data.success && p1Data.pipeline1_result) {
            queryResult.pipeline1_result = {
              success: true,
              sql_query: p1Data.pipeline1_result.sql_query,
              results: p1Data.pipeline1_result.results,
              execution_time: p1Data.pipeline1_result.execution_time,
              metrics: p1Data.pipeline1_result.metrics || {}
            };
            batchResults.summary.pipeline1.success++;
            batchResults.summary.pipeline1.total_time += p1Data.pipeline1_result.execution_time;
            
            addLog(`📝 Generated SQL: ${p1Data.pipeline1_result.sql_query}`, 'sql');
            addLog(`📊 SQL Execution Results: ${p1Data.pipeline1_result.results.length} rows returned`, 'result');
            addLog(`⏱️ Pipeline 1 Total Time: ${(p1Data.pipeline1_result.execution_time * 1000).toFixed(1)}ms`, 'timing');
            
            // Research Metrics Logging
            if (p1Data.pipeline1_result.execution_accuracy !== undefined) {
              addLog(`🎯 Execution Accuracy (EX): ${(p1Data.pipeline1_result.execution_accuracy * 100).toFixed(1)}%`, 'metric');
            }
            if (p1Data.pipeline1_result.latency_ms !== undefined) {
              addLog(`⚡ Latency: ${p1Data.pipeline1_result.latency_ms.toFixed(1)}ms`, 'metric');
            }
            if (p1Data.pipeline1_result.gpu_cost) {
              addLog(`💾 GPU Cost: ${p1Data.pipeline1_result.gpu_cost.gpu_memory_mb.toFixed(1)}MB, ${p1Data.pipeline1_result.gpu_cost.gpu_seconds.toFixed(3)}s`, 'metric');
            }
            
            addLog(`✅ Pipeline 1 Success`, 'success');
          } else {
            queryResult.pipeline1_result = {
              success: false,
              error: p1Data.error || 'Unknown error'
            };
            batchResults.summary.pipeline1.failed++;
            // Error Typology Logging
            if (p1Data.pipeline1_result && p1Data.pipeline1_result.error_type) {
              addLog(`🏷️ Error Type: ${p1Data.pipeline1_result.error_type}`, 'error');
            }
            addLog(`❌ Pipeline 1 Error: ${p1Data.error || 'Unknown error'}`, 'error');
          }
        } catch (error) {
          queryResult.pipeline1_result = {
            success: false,
            error: error.message
          };
          batchResults.summary.pipeline1.failed++;
          addLog(`❌ Pipeline 1 Network Error: ${error.message}`, 'error');
        }
        
        // Execute Pipeline 2
        try {
          addLog(`📡 Pipeline 2 API Request: POST /api/search`, 'api');
          addLog(`📤 Request Body: {"query": "${query.vietnamese}", "pipeline": "pipeline2"}`, 'request');
          
          const p2Response = await fetch('http://localhost:8000/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              query: query.vietnamese, 
              pipeline: 'pipeline2' 
            })
          });
          const p2Data = await p2Response.json();
          
          addLog(`📥 Pipeline 2 Response: ${p2Response.status} ${p2Response.statusText}`, 'response');
          addLog(`📋 Response Data: ${JSON.stringify(p2Data, null, 2)}`, 'response');
          
          if (p2Data.success && p2Data.pipeline2_result) {
            queryResult.pipeline2_result = {
              success: true,
              english_query: p2Data.pipeline2_result.english_query,
              sql_query: p2Data.pipeline2_result.sql_query,
              results: p2Data.pipeline2_result.results,
              execution_time: p2Data.pipeline2_result.execution_time,
              vn_en_time: p2Data.pipeline2_result.vn_en_time,
              en_sql_time: p2Data.pipeline2_result.en_sql_time,
              metrics: p2Data.pipeline2_result.metrics || {}
            };
            batchResults.summary.pipeline2.success++;
            batchResults.summary.pipeline2.total_time += p2Data.pipeline2_result.execution_time;
            
            addLog(`🌐 Vietnamese → English Translation: "${query.vietnamese}" → "${p2Data.pipeline2_result.english_query}"`, 'translation');
            addLog(`📝 Generated SQL: ${p2Data.pipeline2_result.sql_query}`, 'sql');
            addLog(`📊 SQL Execution Results: ${p2Data.pipeline2_result.results.length} rows returned`, 'result');
            addLog(`⏱️ Timing Breakdown: VN→EN ${(p2Data.pipeline2_result.vn_en_time * 1000).toFixed(1)}ms, EN→SQL ${(p2Data.pipeline2_result.en_sql_time * 1000).toFixed(1)}ms`, 'timing');
            addLog(`⏱️ Pipeline 2 Total Time: ${(p2Data.pipeline2_result.execution_time * 1000).toFixed(1)}ms`, 'timing');
            
            // Research Metrics Logging
            if (p2Data.pipeline2_result.execution_accuracy !== undefined) {
              addLog(`🎯 Execution Accuracy (EX): ${(p2Data.pipeline2_result.execution_accuracy * 100).toFixed(1)}%`, 'metric');
            }
            if (p2Data.pipeline2_result.exact_match !== undefined) {
              addLog(`🔍 Exact Match (EM): ${p2Data.pipeline2_result.exact_match ? 'YES' : 'NO'}`, 'metric');
            }
            if (p2Data.pipeline2_result.latency_ms !== undefined) {
              addLog(`⚡ Latency: ${p2Data.pipeline2_result.latency_ms.toFixed(1)}ms`, 'metric');
            }
            if (p2Data.pipeline2_result.gpu_cost) {
              addLog(`💾 GPU Cost: ${p2Data.pipeline2_result.gpu_cost.gpu_memory_mb.toFixed(1)}MB, ${p2Data.pipeline2_result.gpu_cost.gpu_seconds.toFixed(3)}s`, 'metric');
            }
            
            addLog(`✅ Pipeline 2 Success`, 'success');
          } else {
            queryResult.pipeline2_result = {
              success: false,
              error: p2Data.error || 'Unknown error'
            };
            batchResults.summary.pipeline2.failed++;
            // Error Typology Logging
        if (p2Data.pipeline2_result && p2Data.pipeline2_result.error_type) {
          addLog(`🏷️ Error Type: ${p2Data.pipeline2_result.error_type}`, 'error');
        }
        addLog(`❌ Pipeline 2 Error: ${p2Data.error || 'Unknown error'}`, 'error');
          }
        } catch (error) {
          queryResult.pipeline2_result = {
            success: false,
            error: error.message
          };
          batchResults.summary.pipeline2.failed++;
          addLog(`❌ Pipeline 2 Network Error: ${error.message}`, 'error');
        }
        
        batchResults.results.push(queryResult);
        
        // Add small delay between queries to avoid overwhelming the API
        if (i < allQueries.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }
      
      // Save results to backend
      try {
        const saveResponse = await fetch('http://localhost:8000/api/batch-results/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: filename,
            data: batchResults
          })
        });
        
        if (saveResponse.ok) {
          addLog(`💾 Results saved to: ${filename}`, 'success');
        } else {
          addLog(`⚠️ Failed to save results file`, 'error');
        }
      } catch (saveError) {
        addLog(`⚠️ Save error: ${saveError.message}`, 'error');
      }
      
      setQueryResults(batchResults);
      
      // Generate complexity report
      const complexityReport = generateComplexityReport(batchResults);
      setComplexityReport(complexityReport);
      
      addLog(`🚀 Starting optimized batch execution for ${selectedComplexity} queries...`, 'info');
      
      // Use new batch execution endpoint
      const batchResponse = await fetch('http://localhost:8000/api/sample-queries/execute-samples-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complexity: selectedComplexity,
          pipeline: 'both'
        })
      });
      
      if (!batchResponse.ok) {
        throw new Error(`Batch execution failed: ${batchResponse.statusText}`);
      }
      
      const batchData = await batchResponse.json();
      
      addLog(`✅ Batch execution completed!`, 'success');
      addLog(`📊 Total batches processed: ${batchData.total_batches}`, 'info');
      addLog(`📊 Total queries processed: ${batchData.total_queries}`, 'info');
      addLog(`⏱️ Total execution time: ${batchData.total_execution_time.toFixed(2)}s`, 'info');
      
      const p1Success = complexityResults.filter(r => r.pipeline1_result?.success).length;
      const p2Success = complexityResults.filter(r => r.pipeline2_result?.success).length;
      
      const p1AvgTime = complexityResults
        .filter(r => r.pipeline1_result?.success)
        .reduce((sum, r) => sum + (r.pipeline1_result.execution_time || 0), 0) / (p1Success || 1) * 1000;
        
      const p2AvgTime = complexityResults
        .filter(r => r.pipeline2_result?.success)
        .reduce((sum, r) => sum + (r.pipeline2_result.execution_time || 0), 0) / (p2Success || 1) * 1000;
      
      addLog(`🎉 Batch execution completed successfully!`, 'success');
      addLog(`📊 Pipeline 1: ${batchResults.summary.pipeline1.success}/${allQueries.length} success`, 'info');
      addLog(`📊 Pipeline 2: ${batchResults.summary.pipeline2.success}/${allQueries.length} success`, 'info');
      
    } catch (error) {
      addLog(`❌ Batch execution failed: ${error.message}`, 'error');
      console.error('Error executing queries:', error);
    }
    setLoading(false);
  };
  
  const executeQueries = async () => {
    if (!selectedComplexity) {
      alert('Please select a complexity level');
      return;
    }
    
    setLoading(true);
    setQueryResults(null);
    setComplexityReport(null);
    setExecutionLogs([]);
    
    const addLog = (message, type = 'info') => {
      const timestamp = new Date().toLocaleTimeString();
      setExecutionLogs(prev => [...prev, { timestamp, message, type }]);
    };
    
    try {
      addLog(`🚀 Starting optimized batch execution for ${selectedComplexity} queries...`, 'info');
      
      const batchResponse = await fetch('http://localhost:8000/api/sample-queries/execute-samples-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complexity: selectedComplexity,
          pipeline: 'both'
        })
      });
      
      if (!batchResponse.ok) {
        throw new Error(`Batch execution failed: ${batchResponse.statusText}`);
      }
      
      const batchData = await batchResponse.json();
      
      addLog(`✅ Batch execution completed!`, 'success');
      addLog(`📊 Total batches: ${batchData.total_batches}`, 'info');
      addLog(`📊 Total queries: ${batchData.total_queries}`, 'info');
      addLog(`⏱️ Total time: ${batchData.total_execution_time.toFixed(2)}s`, 'info');
      
      const processedResults = {
        complexity: batchData.complexity,
        total_queries: batchData.total_queries,
        results: [],
        summary: {
          pipeline1: {
            success: batchData.overall_stats.pipeline1.success,
            failed: batchData.overall_stats.pipeline1.errors,
            total_time: batchData.overall_stats.pipeline1.total_time
          },
          pipeline2: {
            success: batchData.overall_stats.pipeline2.success,
            failed: batchData.overall_stats.pipeline2.errors,
            total_time: batchData.overall_stats.pipeline2.total_time
          }
        },
        execution_timestamp: new Date().toISOString()
      };
      
      setQueryResults(processedResults);
      
      const complexityReport = generateComplexityReport(processedResults);
      setComplexityReport(complexityReport);
      
      addLog(`🎉 Batch execution completed successfully!`, 'success');
      addLog(`📊 P1: ${processedResults.summary.pipeline1.success}/${batchData.total_queries}`, 'info');
      addLog(`📊 P2: ${processedResults.summary.pipeline2.success}/${batchData.total_queries}`, 'info');
      
    } catch (error) {
      addLog(`❌ Batch execution failed: ${error.message}`, 'error');
      console.error('Error executing queries:', error);
    }
    setLoading(false);
  };
  
  const generateComplexityReport = (batchResults) => {
    const complexities = ['simple', 'medium', 'complex'];
    const reports = complexities.map(complexity => {
      const complexityResults = batchResults.results.filter(r => r.complexity === complexity);
      const totalQueries = complexityResults.length;
      
      if (totalQueries === 0) return null;
      
      const p1Success = complexityResults.filter(r => r.pipeline1_result?.success).length;
      const p2Success = complexityResults.filter(r => r.pipeline2_result?.success).length;
      
      const p1AvgTime = complexityResults
        .filter(r => r.pipeline1_result?.success)
        .reduce((sum, r) => sum + (r.pipeline1_result.execution_time || 0), 0) / (p1Success || 1) * 1000;
        
      const p2AvgTime = complexityResults
        .filter(r => r.pipeline2_result?.success)
        .reduce((sum, r) => sum + (r.pipeline2_result.execution_time || 0), 0) / (p2Success || 1) * 1000;
      
      return {
        complexity,
        total_queries: totalQueries,
        pipeline1_stats: {
          success_count: p1Success,
          success_rate: (p1Success / totalQueries) * 100,
          average_time_ms: p1AvgTime
        },
        pipeline2_stats: {
          success_count: p2Success,
          success_rate: (p2Success / totalQueries) * 100,
          average_time_ms: p2AvgTime
        },
        comparison: {
          faster_pipeline: p1AvgTime < p2AvgTime ? 'Pipeline 1' : 'Pipeline 2',
          sql_exact_match_rate: 0 // Would need to implement SQL comparison logic
        }
      };
    }).filter(Boolean);
    
    return { complexity_reports: reports };
  };

  const clearLogs = () => {
    setExecutionLogs([]);
    addLog('🧹 Logs cleared', 'info');
  };

  const filteredResults = queryResults?.results?.filter(result => {
    const complexityMatch = selectedComplexity === 'all' || result.complexity === selectedComplexity;
    return complexityMatch;
  }) || [];

  const getFilteredSampleQueries = () => {
    if (selectedComplexity === 'all') {
      return Object.values(sampleQueriesData).flat();
    }
    return sampleQueriesData[selectedComplexity] || [];
  };

  const complexityLevels = [
    { value: 'all', label: 'All Complexities', count: Object.values(sampleQueriesData).flat().length },
    { value: 'simple', label: 'Simple Queries', count: sampleQueriesData.simple.length },
    { value: 'medium', label: 'Medium Queries', count: sampleQueriesData.medium.length },
    { value: 'complex', label: 'Complex Queries', count: sampleQueriesData.complex.length }
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
      case 'api': return '📡';
      case 'request': return '📤';
      case 'response': return '📥';
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
        <title>Sample Queries - Vietnamese NL2SQL System</title>
        <meta name="description" content="Interactive sample queries for Vietnamese to SQL translation testing" />
      </Head>

      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.titleSection}>
            <h1>Vietnamese NL2SQL Sample Queries</h1>
            <p>Interactive testing environment for Vietnamese natural language to SQL translation</p>
            <div className={styles.statsBar}>
              <div className={styles.stat}>
                <span className={styles.statNumber}>{Object.values(sampleQueriesData).flat().length}</span>
                <span className={styles.statLabel}>Vietnamese-SQL Pairs</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>2</span>
                <span className={styles.statLabel}>Pipelines</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>{executionLogs.length}</span>
                <span className={styles.statLabel}>Log Entries</span>
              </div>
            </div>
          </div>
          
          <div className={`${styles.controls} ${loading ? styles.frozen : ''}`}>
            <div className={styles.controlGroup}>
              <label htmlFor="complexity">Complexity Level:</label>
              <select 
                id="complexity"
                value={selectedComplexity} 
                onChange={(e) => setSelectedComplexity(e.target.value)}
                className={styles.filterSelect}
                disabled={loading}
              >
                {complexityLevels.map(level => (
                  <option key={level.value} value={level.value}>
                    {level.label} ({level.count})
                  </option>
                ))}
              </select>
            </div>
            
            <button 
              onClick={executeAllQueries} 
              disabled={loading}
              className={`${styles.executeButton} ${loading ? styles.loading : ''}`}
            >
              {loading ? (
                <>
                  <span className={styles.spinner}></span>
                  Processing...
                </>
              ) : (
                '🚀 Execute All Queries'
              )}
            </button>
          </div>
        </div>

        {/* Execution Logs Section - Full Width Under Banner */}
        <div className={styles.logsSection}>
          <div className={styles.sectionHeader}>
            <div className={styles.logHeader}>
              <h2>📋 Execution Logs</h2>
              <div className={styles.logControls}>
                <button 
                  onClick={() => setShowLogs(!showLogs)}
                  className={styles.toggleButton}
                >
                  {showLogs ? '🙈 Hide' : '👁️ Show'}
                </button>
                <button 
                  onClick={clearLogs}
                  className={styles.clearButton}
                >
                  🧹 Clear
                </button>
              </div>
            </div>
            <p>Real-time logging of pipeline execution and results</p>
          </div>
          
          {showLogs && (
            <div className={styles.logsContainer}>
              {executionLogs.length === 0 ? (
                <div className={styles.emptyLogs}>
                  <p>No logs yet. Execute a query to see detailed logging information.</p>
                </div>
              ) : (
                <div className={styles.logsList}>
                  {executionLogs.map((log) => (
                    <div key={log.id} className={`${styles.logEntry} ${styles[log.type]}`}>
                      <span className={styles.logTimestamp}>{log.timestamp}</span>
                      <span className={styles.logIcon}>{getLogIcon(log.type)}</span>
                      <span className={styles.logMessage}>{log.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className={styles.mainContent}>
          {/* Sample Queries Section - Full Width Below Logs */}
          <div className={styles.sampleQueriesSection}>
            <div className={styles.sectionHeader}>
              <h2>📝 Sample Queries ({getFilteredSampleQueries().length})</h2>
              <p>Click any query to test it on both pipelines</p>
            </div>
            
            <div className={styles.queriesGrid}>
              {getFilteredSampleQueries().map((query, index) => (
                <div key={index} className={styles.queryCard}>
                  <div className={styles.queryCardHeader}>
                    <span 
                      className={styles.complexityBadge}
                      style={{ backgroundColor: getComplexityColor(query.complexity) }}
                    >
                      {query.complexity.toUpperCase()}
                    </span>
                  </div>
                  <div className={styles.queryContent}>
                    <div className={styles.vietnameseQuery}>
                      <strong>🇻🇳 Vietnamese:</strong> "{query.vietnamese}"
                    </div>
                    <div className={styles.englishQuery}>
                      <strong>🇺🇸 English:</strong> "{query.english}"
                    </div>
                    {query.sql && (
                      <div className={styles.sqlQuery}>
                        <strong>🗄️ SQL:</strong>
                        <pre className={styles.sqlCode}>{query.sql}</pre>
                      </div>
                    )}
                  </div>
                  <div className={styles.queryActions}>
                    <button 
                      onClick={() => executeQuery(query.vietnamese, 'both')}
                      className={styles.testButton}
                      disabled={loading}
                    >
                      🔄 Test Both Pipelines
                    </button>
                    <button 
                      onClick={() => executeQuery(query.vietnamese, '1')}
                      className={styles.testButtonSmall}
                      disabled={loading}
                    >
                      P1
                    </button>
                    <button 
                      onClick={() => executeQuery(query.vietnamese, '2')}
                      className={styles.testButtonSmall}
                      disabled={loading}
                    >
                      P2
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>


        {complexityReport && (
          <div className={styles.reportSection}>
            <div className={styles.sectionHeader}>
              <h2>📊 Performance Report</h2>
              <p>Comprehensive analysis of pipeline performance by complexity</p>
            </div>
            <div className={styles.reportGrid}>
              {complexityReport.complexity_reports.map((report) => (
                <div key={report.complexity} className={styles.complexityCard}>
                  <h3 style={{ color: getComplexityColor(report.complexity) }}>
                    {report.complexity.toUpperCase()} Queries
                  </h3>
                  <div className={styles.statsGrid}>
                    <div className={styles.pipelineStats}>
                      <h4>Pipeline 1</h4>
                      <p>Success: {report.pipeline1_stats.success_count}/{report.total_queries}</p>
                      <p>Rate: {report.pipeline1_stats.success_rate.toFixed(1)}%</p>
                      <p>Avg Time: {formatTime(report.pipeline1_stats.average_time_ms)}</p>
                    </div>
                    <div className={styles.pipelineStats}>
                      <h4>Pipeline 2</h4>
                      <p>Success: {report.pipeline2_stats.success_count}/{report.total_queries}</p>
                      <p>Rate: {report.pipeline2_stats.success_rate.toFixed(1)}%</p>
                      <p>Avg Time: {formatTime(report.pipeline2_stats.average_time_ms)}</p>
                    </div>
                  </div>
                  <div className={styles.comparison}>
                    <p><strong>Winner:</strong> {report.comparison.faster_pipeline}</p>
                    <p><strong>SQL Match Rate:</strong> {report.comparison.sql_exact_match_rate.toFixed(1)}%</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {queryResults && (
          <div className={styles.resultsSection}>
            <div className={styles.sectionHeader}>
              <h2>🔍 Batch Results ({filteredResults.length} queries)</h2>
              <p>Results from comprehensive batch execution</p>
            </div>
            
            <div className={styles.queryGrid}>
              {filteredResults.map((result) => (
                <div key={result.query_id} className={styles.queryCard}>
                  <div className={styles.queryHeader}>
                    <span 
                      className={styles.complexityBadge}
                      style={{ backgroundColor: getComplexityColor(result.complexity) }}
                    >
                      {result.complexity}
                    </span>
                    <span className={styles.queryId}>#{result.query_id}</span>
                  </div>
                  
                  <div className={styles.queryContent}>
                    <h4>"{result.vietnamese_query}"</h4>
                    <p className={styles.challenge}><strong>Challenge:</strong> {result.challenge}</p>
                    <p className={styles.expectedSql}><strong>Expected:</strong> {result.expected_sql_type}</p>
                  </div>

                  <div className={styles.pipelineResults}>
                    {/* Pipeline 1 Results */}
                    <div className={styles.pipelineResult}>
                      <h5>Pipeline 1 Result</h5>
                      {result.pipeline1_result?.success ? (
                        <div className={styles.successResult}>
                          <p className={styles.sqlQuery}>
                            <strong>SQL:</strong> {result.pipeline1_result.sql_query}
                          </p>
                          <p className={styles.resultCount}>
                            Results: {result.pipeline1_result.result_count} | 
                            Time: {formatTime(result.pipeline1_result.execution_time * 1000)}
                          </p>
                          {result.pipeline1_result.results.length > 0 && (
                            <button 
                              onClick={() => setSelectedQuery(result.pipeline1_result.results)}
                              className={styles.viewResultsBtn}
                            >
                              View Sample Results
                            </button>
                          )}
                        </div>
                      ) : (
                        <div className={styles.errorResult}>
                          <p className={styles.error}>Error: {result.pipeline1_result?.error || 'Unknown error'}</p>
                        </div>
                      )}
                    </div>

                    {/* Pipeline 2 Results */}
                    <div className={styles.pipelineResult}>
                      <h5>Pipeline 2 Result</h5>
                      {result.pipeline2_result?.success ? (
                        <div className={styles.successResult}>
                          <p className={styles.englishQuery}>
                            <strong>English:</strong> {result.pipeline2_result.english_query}
                          </p>
                          <p className={styles.sqlQuery}>
                            <strong>SQL:</strong> {result.pipeline2_result.sql_query}
                          </p>
                          <p className={styles.resultCount}>
                            Results: {result.pipeline2_result.result_count} | 
                            Time: {formatTime(result.pipeline2_result.execution_time * 1000)} |
                            VN→EN: {formatTime(result.pipeline2_result.vn_en_time * 1000)} |
                            EN→SQL: {formatTime(result.pipeline2_result.en_sql_time * 1000)}
                          </p>
                          {result.pipeline2_result.results.length > 0 && (
                            <button 
                              onClick={() => setSelectedQuery(result.pipeline2_result.results)}
                              className={styles.viewResultsBtn}
                            >
                              View Sample Results
                            </button>
                          )}
                        </div>
                      ) : (
                        <div className={styles.errorResult}>
                          <p className={styles.error}>Error: {result.pipeline2_result?.error || 'Unknown error'}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* SQL Comparison */}
                  {result.pipeline1_result?.success && result.pipeline2_result?.success && (
                    <div className={styles.comparison}>
                      <p className={styles.sqlMatch}>
                        <strong>SQL Match:</strong> {
                          result.pipeline1_result.sql_query.trim() === result.pipeline2_result.sql_query.trim() 
                            ? '✅ Identical' : '❌ Different'
                        }
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results Modal */}
        {selectedQuery && (
          <div className={styles.modal} onClick={() => setSelectedQuery(null)}>
            <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
              <h3>Sample Query Results</h3>
              <button 
                className={styles.closeBtn}
                onClick={() => setSelectedQuery(null)}
              >
                ×
              </button>
              <div className={styles.resultsTable}>
                {selectedQuery.length > 0 ? (
                  <table>
                    <thead>
                      <tr>
                        {Object.keys(selectedQuery[0]).map(key => (
                          <th key={key}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {selectedQuery.map((row, idx) => (
                        <tr key={idx}>
                          {Object.values(row).map((value, i) => (
                            <td key={i}>{String(value)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p>No results found</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
