-- Business question: how many restaurants are low/medium/high popularity by
-- review count, and does popularity track with average rating -- are the
-- most-reviewed restaurants also the best-rated, or is popularity mostly
-- just review volume? Thresholds from the business review_count distribution
-- (25th percentile 14, median 34, 90th percentile 195).

CREATE OR REPLACE VIEW restaurant_segments AS
SELECT
    business_id,
    name,
    review_count,
    stars,
    CASE
        WHEN review_count < 20 THEN 'low'
        WHEN review_count < 100 THEN 'medium'
        ELSE 'high'
    END AS popularity_tier
FROM business;

CREATE OR REPLACE VIEW restaurant_segment_summary AS
SELECT
    popularity_tier,
    COUNT(*) AS n_restaurants,
    ROUND(AVG(stars), 2) AS avg_stars,
    ROUND(AVG(review_count), 1) AS avg_review_count
FROM restaurant_segments
GROUP BY popularity_tier
ORDER BY CASE popularity_tier WHEN 'low' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END;
