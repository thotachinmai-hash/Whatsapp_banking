-- One-off update for an already-running database whose loan_products rows
-- still have the old, unrealistically low min/max amount and tenure
-- ranges from before this change. infra/postgres/init.sql only runs on a
-- fresh install, so a live DB needs this run once by hand:
--
--   psql "$DATABASE_URL" -f infra/postgres/update_loan_products.sql
--
-- Safe to run more than once (idempotent — always sets the same values).

UPDATE loan_products SET min_amount = 10000.00,  max_amount = 1000000.00, min_tenure_months = 2,  max_tenure_months = 60 WHERE loan_type = 'personal';
UPDATE loan_products SET min_amount = 100000.00, max_amount = 4000000.00, min_tenure_months = 12, max_tenure_months = 78 WHERE loan_type = 'home';
UPDATE loan_products SET min_amount = 20000.00,  max_amount = 1500000.00, min_tenure_months = 2,  max_tenure_months = 60 WHERE loan_type = 'vehicle';
UPDATE loan_products SET min_amount = 10000.00,  max_amount = 2000000.00, min_tenure_months = 2,  max_tenure_months = 78 WHERE loan_type = 'education';
