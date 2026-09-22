-- Business question: how many users are new (1-2 reviews), casual (3-9), or
-- power users (10+ reviews), and how much of total review volume does each
-- group drive? Thresholds come from the review-count distribution (median 1,
-- 90th percentile 6, 95th percentile 11 reviews/user) -- most users are
-- one-and-done reviewers, and a long tail of power users drives outsized
-- volume. This is also the exact segment boundary that determines whether a
-- collaborative-filtering / matrix-factorization model can serve a user at
-- all: those models only train on users with >=5 reviews (see
-- src/recsys/build_data.py's 5-core filter), so "new" and part of "casual"
-- are structurally cold-start for them -- the segment failure the brief asks
-- about is visible here before a single model is even trained.

CREATE OR REPLACE VIEW user_review_counts AS
SELECT user_id, COUNT(*) AS n_reviews
FROM reviews_all
GROUP BY user_id;

CREATE OR REPLACE VIEW user_segments AS
SELECT
    user_id,
    n_reviews,
    CASE
        WHEN n_reviews <= 2 THEN 'new'
        WHEN n_reviews <= 9 THEN 'casual'
        ELSE 'power'
    END AS user_segment
FROM user_review_counts;

CREATE OR REPLACE VIEW user_segment_summary AS
SELECT
    user_segment,
    COUNT(*) AS n_users,
    SUM(n_reviews) AS total_reviews,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_users,
    ROUND(100.0 * SUM(n_reviews) / SUM(SUM(n_reviews)) OVER (), 2) AS pct_of_review_volume
FROM user_segments
GROUP BY user_segment
ORDER BY CASE user_segment WHEN 'new' THEN 1 WHEN 'casual' THEN 2 ELSE 3 END;
