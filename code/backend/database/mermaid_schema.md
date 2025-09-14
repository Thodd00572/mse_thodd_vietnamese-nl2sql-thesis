# Vietnamese Tiki Products Database Schema

<!-- Mermaid ERD Code (stored for reference, not displayed in frontend) -->

```mermaid
erDiagram
    brands {
        int brand_id PK
        text brand_name
        int product_count
    }
    
    categories {
        int category_id PK
        text category_name
        int product_count
    }
    
    sellers {
        int seller_id PK
        text seller_name
        int product_count
        int total_quantity_sold
        real avg_rating
    }
    
    products {
        int product_id PK
        int tiki_id
        text name
        text description
        int brand_id FK
        int category_id FK
        int seller_id FK
        int date_created
        int number_of_images
    }
    
    product_pricing {
        int pricing_id PK
        int product_id FK
        real price
        real original_price
        real discount_rate
        int quantity_sold
        int favourite_count
        boolean pay_later
        int vnd_cashback
    }
    
    product_reviews {
        int review_id PK
        int product_id FK
        real rating_average
        int review_count
        boolean has_video
    }
    
    brands ||--o{ products : "brand_id"
    categories ||--o{ products : "category_id"
    sellers ||--o{ products : "seller_id"
    products ||--|| product_pricing : "product_id"
    products ||--|| product_reviews : "product_id"
```

## Schema Image

The visual ERD diagram is now displayed as an image in the frontend at:
`/code/frontend/public/images/tiki_database_schema.png`

## Table Relationships

- **brands** → **products**: One brand can have many products
- **categories** → **products**: One category can have many products  
- **sellers** → **products**: One seller can have many products
- **products** → **product_pricing**: One-to-one relationship for pricing data
- **products** → **product_reviews**: One-to-one relationship for review data

## Key Features

- **Normalized Structure**: 6 tables with proper foreign key relationships
- **Complex JOIN Support**: Enables multi-table Vietnamese NL2SQL queries
- **Data Integrity**: Foreign key constraints ensure referential integrity
- **Performance Optimized**: Indexed foreign keys for fast JOIN operations
- **Vietnamese E-commerce**: Tailored for Tiki marketplace product data

## Record Counts

- **brands**: 824 unique brands
- **categories**: 155 product categories
- **sellers**: 3,807 marketplace sellers
- **products**: 41,576 core product records
- **product_pricing**: 83,206 pricing records
- **product_reviews**: 83,206 review records
