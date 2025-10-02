"""
SQL Templates for Vietnamese E-commerce NL2SQL Pipeline 1
Author: MSE14 Duong Dinh Tho
"""

from typing import Dict, List, Any
import re
from enum import Enum

class QueryIntent(Enum):
    """Query intent categories for Vietnamese e-commerce queries"""
    PRODUCT_SEARCH = "product_search"
    PRICE_FILTER = "price_filter"
    BRAND_FILTER = "brand_filter"
    CATEGORY_FILTER = "category_filter"
    RATING_FILTER = "rating_filter"
    SORT_PRODUCTS = "sort_products"
    COUNT_PRODUCTS = "count_products"
    COMPARISON = "comparison"
    TOP_PRODUCTS = "top_products"

class SQLTemplateBuilder:
    """Builds SQL queries from Vietnamese intents and entities"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.vietnamese_keywords = self._initialize_keywords()
        
    def _initialize_keywords(self) -> Dict[str, List[str]]:
        """Vietnamese keywords for entity extraction"""
        return {
            # Price keywords
            "price_low": ["rẻ", "giá rẻ", "giá thấp", "dưới", "nhỏ hơn", "ít hơn", "không quá"],
            "price_high": ["đắt", "giá cao", "trên", "lớn hơn", "nhiều hơn", "từ"],
            "price_range": ["từ", "đến", "trong khoảng", "giữa"],
            
            # Sort keywords
            "sort_price_asc": ["giá tăng dần", "giá từ thấp đến cao", "rẻ nhất"],
            "sort_price_desc": ["giá giảm dần", "giá từ cao đến thấp", "đắt nhất"],
            "sort_rating": ["đánh giá cao", "rating cao", "tốt nhất"],
            "sort_popular": ["bán chạy", "phổ biến", "nhiều người mua"],
            
            # Category keywords
            "categories": {
                "điện thoại": ["điện thoại", "smartphone", "phone", "di động"],
                "laptop": ["laptop", "máy tính xách tay", "notebook"],
                "túi xách": ["túi xách", "túi", "balo", "cặp"],
                "thời trang": ["quần áo", "thời trang", "áo", "quần"],
                "phụ kiện": ["phụ kiện", "cài áo", "trang sức"]
            },
            
            # Brand keywords (common Vietnamese brands)
            "brands": ["Samsung", "Apple", "Xiaomi", "Oppo", "Vivo", "Sony", "LG", "Huawei"],
            
            # Count/Top keywords
            "count": ["có bao nhiêu", "số lượng", "tổng cộng"],
            "top": ["top", "hàng đầu", "tốt nhất", "cao nhất", "nhiều nhất"]
        }
    
    def _initialize_templates(self) -> Dict[QueryIntent, Dict[str, str]]:
        """SQL templates for each query intent"""
        return {
            QueryIntent.PRODUCT_SEARCH: {
                "basic": "SELECT * FROM products WHERE name LIKE '%{product_name}%'",
                "with_category": "SELECT * FROM products WHERE name LIKE '%{product_name}%' AND category = '{category}'",
                "with_brand": "SELECT * FROM products WHERE name LIKE '%{product_name}%' AND brand = '{brand}'"
            },
            
            QueryIntent.PRICE_FILTER: {
                "under": "SELECT * FROM products WHERE price <= {max_price}",
                "over": "SELECT * FROM products WHERE price >= {min_price}",
                "range": "SELECT * FROM products WHERE price BETWEEN {min_price} AND {max_price}",
                "with_category": "SELECT * FROM products WHERE price <= {max_price} AND category = '{category}'"
            },
            
            QueryIntent.BRAND_FILTER: {
                "basic": "SELECT * FROM products WHERE brand = '{brand}'",
                "with_category": "SELECT * FROM products WHERE brand = '{brand}' AND category = '{category}'",
                "with_price": "SELECT * FROM products WHERE brand = '{brand}' AND price <= {max_price}"
            },
            
            QueryIntent.CATEGORY_FILTER: {
                "basic": "SELECT * FROM products WHERE category = '{category}'",
                "with_price": "SELECT * FROM products WHERE category = '{category}' AND price <= {max_price}",
                "with_rating": "SELECT * FROM products WHERE category = '{category}' AND rating_average >= {min_rating}"
            },
            
            QueryIntent.RATING_FILTER: {
                "basic": "SELECT * FROM products WHERE rating_average >= {min_rating}",
                "with_category": "SELECT * FROM products WHERE rating_average >= {min_rating} AND category = '{category}'",
                "with_price": "SELECT * FROM products WHERE rating_average >= {min_rating} AND price <= {max_price}"
            },
            
            QueryIntent.SORT_PRODUCTS: {
                "price_asc": "SELECT * FROM products ORDER BY price ASC",
                "price_desc": "SELECT * FROM products ORDER BY price DESC",
                "rating_desc": "SELECT * FROM products ORDER BY rating_average DESC",
                "popular": "SELECT * FROM products ORDER BY quantity_sold DESC",
                "with_category": "SELECT * FROM products WHERE category = '{category}' ORDER BY {sort_field} {sort_order}"
            },
            
            QueryIntent.COUNT_PRODUCTS: {
                "basic": "SELECT COUNT(*) as total_products FROM products",
                "by_category": "SELECT COUNT(*) as total_products FROM products WHERE category = '{category}'",
                "by_brand": "SELECT COUNT(*) as total_products FROM products WHERE brand = '{brand}'",
                "by_price": "SELECT COUNT(*) as total_products FROM products WHERE price <= {max_price}"
            },
            
            QueryIntent.TOP_PRODUCTS: {
                "by_rating": "SELECT * FROM products ORDER BY rating_average DESC LIMIT {limit}",
                "by_sales": "SELECT * FROM products ORDER BY quantity_sold DESC LIMIT {limit}",
                "by_price_low": "SELECT * FROM products ORDER BY price ASC LIMIT {limit}",
                "by_category": "SELECT * FROM products WHERE category = '{category}' ORDER BY rating_average DESC LIMIT {limit}"
            },
            
            QueryIntent.COMPARISON: {
                "brands": "SELECT brand, AVG(price) as avg_price, AVG(rating_average) as avg_rating FROM products WHERE brand IN ({brands}) GROUP BY brand",
                "categories": "SELECT category, COUNT(*) as product_count, AVG(price) as avg_price FROM products GROUP BY category"
            }
        }
    
    def extract_entities(self, vietnamese_query: str) -> Dict[str, Any]:
        """Extract entities from Vietnamese query using keyword matching"""
        entities = {}
        query_lower = vietnamese_query.lower()
        
        # Extract price values
        price_pattern = r'(\d+(?:\.\d+)?)\s*(?:k|nghìn|triệu|tr|đồng|vnd)?'
        prices = re.findall(price_pattern, query_lower)
        if prices:
            # Convert to actual values (handle k, triệu)
            price_values = []
            for price in prices:
                value = float(price)
                if 'k' in query_lower or 'nghìn' in query_lower:
                    value *= 1000
                elif 'triệu' in query_lower or 'tr' in query_lower:
                    value *= 1000000
                price_values.append(value)
            entities['prices'] = price_values
        
        # Extract categories
        for category, keywords in self.vietnamese_keywords['categories'].items():
            for keyword in keywords:
                if keyword in query_lower:
                    entities['category'] = category
                    break
        
        # Extract brands
        for brand in self.vietnamese_keywords['brands']:
            if brand.lower() in query_lower:
                entities['brand'] = brand
                break
        
        # Extract rating
        rating_pattern = r'(\d+(?:\.\d+)?)\s*(?:sao|star|rating)'
        ratings = re.findall(rating_pattern, query_lower)
        if ratings:
            entities['rating'] = float(ratings[0])
        
        # Extract limit for top queries
        limit_pattern = r'(\d+)\s*(?:sản phẩm|sp|item)'
        limits = re.findall(limit_pattern, query_lower)
        if limits:
            entities['limit'] = int(limits[0])
        else:
            entities['limit'] = 10  # default
        
        return entities
    
    def classify_intent(self, vietnamese_query: str) -> QueryIntent:
        """Classify Vietnamese query intent"""
        query_lower = vietnamese_query.lower()
        
        # Check for count queries
        if any(keyword in query_lower for keyword in self.vietnamese_keywords['count']):
            return QueryIntent.COUNT_PRODUCTS
        
        # Check for top/best queries
        if any(keyword in query_lower for keyword in self.vietnamese_keywords['top']):
            return QueryIntent.TOP_PRODUCTS
        
        # Check for price filters
        if any(keyword in query_lower for keyword in self.vietnamese_keywords['price_low'] + self.vietnamese_keywords['price_high']):
            return QueryIntent.PRICE_FILTER
        
        # Check for sorting
        if any(keyword in query_lower for keyword in self.vietnamese_keywords['sort_price_asc'] + self.vietnamese_keywords['sort_price_desc']):
            return QueryIntent.SORT_PRODUCTS
        
        # Check for rating filters
        if 'sao' in query_lower or 'rating' in query_lower or 'đánh giá' in query_lower:
            return QueryIntent.RATING_FILTER
        
        # Check for brand filters
        if any(brand.lower() in query_lower for brand in self.vietnamese_keywords['brands']):
            return QueryIntent.BRAND_FILTER
        
        # Check for category filters
        if any(any(keyword in query_lower for keyword in keywords) 
               for keywords in self.vietnamese_keywords['categories'].values()):
            return QueryIntent.CATEGORY_FILTER
        
        # Default to product search
        return QueryIntent.PRODUCT_SEARCH
    
    def build_sql(self, vietnamese_query: str) -> Dict[str, Any]:
        """Build SQL query from Vietnamese natural language"""
        
        # Extract entities and classify intent
        entities = self.extract_entities(vietnamese_query)
        intent = self.classify_intent(vietnamese_query)
        
        # Get appropriate template
        template_group = self.templates[intent]
        
        # Select specific template based on available entities
        template_key = self._select_template(intent, entities)
        sql_template = template_group[template_key]
        
        # Fill template with entities
        try:
            sql_query = self._fill_template(sql_template, entities)
            
            return {
                "sql_query": sql_query,
                "intent": intent.value,
                "entities": entities,
                "template_used": template_key,
                "success": True
            }
        except Exception as e:
            return {
                "sql_query": None,
                "intent": intent.value,
                "entities": entities,
                "error": str(e),
                "success": False
            }
    
    def _select_template(self, intent: QueryIntent, entities: Dict[str, Any]) -> str:
        """Select most appropriate template based on available entities"""
        
        if intent == QueryIntent.PRODUCT_SEARCH:
            if 'category' in entities:
                return "with_category"
            elif 'brand' in entities:
                return "with_brand"
            return "basic"
        
        elif intent == QueryIntent.PRICE_FILTER:
            if 'category' in entities:
                return "with_category"
            elif len(entities.get('prices', [])) >= 2:
                return "range"
            elif any(keyword in entities for keyword in ['prices']):
                # Determine if it's under or over based on query context
                return "under"  # Default to under for single price
            return "under"
        
        elif intent == QueryIntent.BRAND_FILTER:
            if 'category' in entities:
                return "with_category"
            elif 'prices' in entities:
                return "with_price"
            return "basic"
        
        elif intent == QueryIntent.CATEGORY_FILTER:
            if 'prices' in entities:
                return "with_price"
            elif 'rating' in entities:
                return "with_rating"
            return "basic"
        
        elif intent == QueryIntent.SORT_PRODUCTS:
            if 'category' in entities:
                return "with_category"
            # Determine sort type from query
            return "price_asc"  # Default
        
        elif intent == QueryIntent.COUNT_PRODUCTS:
            if 'category' in entities:
                return "by_category"
            elif 'brand' in entities:
                return "by_brand"
            elif 'prices' in entities:
                return "by_price"
            return "basic"
        
        elif intent == QueryIntent.TOP_PRODUCTS:
            if 'category' in entities:
                return "by_category"
            return "by_rating"  # Default
        
        return "basic"
    
    def _fill_template(self, template: str, entities: Dict[str, Any]) -> str:
        """Fill SQL template with extracted entities"""
        
        # Prepare parameters for template filling
        params = {}
        
        if 'prices' in entities:
            prices = entities['prices']
            if len(prices) >= 2:
                params['min_price'] = min(prices)
                params['max_price'] = max(prices)
            elif len(prices) == 1:
                params['max_price'] = prices[0]
                params['min_price'] = prices[0]
        
        if 'category' in entities:
            params['category'] = entities['category']
        
        if 'brand' in entities:
            params['brand'] = entities['brand']
        
        if 'rating' in entities:
            params['min_rating'] = entities['rating']
        
        if 'limit' in entities:
            params['limit'] = entities['limit']
        
        # Handle product name extraction (simple approach)
        # This would need more sophisticated NER in practice
        params['product_name'] = ""  # Placeholder
        
        # Fill template
        return template.format(**params)

# Example usage and test cases
if __name__ == "__main__":
    builder = SQLTemplateBuilder()
    
    # Test queries
    test_queries = [
        "Tìm điện thoại Samsung dưới 10 triệu",
        "Laptop có giá từ 15 triệu đến 25 triệu",
        "Top 5 sản phẩm bán chạy nhất",
        "Có bao nhiêu túi xách trong cửa hàng?",
        "Sản phẩm Apple có rating trên 4 sao",
        "Sắp xếp theo giá tăng dần"
    ]
    
    for query in test_queries:
        result = builder.build_sql(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
        print(f"SQL: {result['sql_query']}")
        print("-" * 50)
