-- HSBC WhatsApp Banking Assistant — Database Schema

CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    account_number VARCHAR(30) UNIQUE NOT NULL,
    account_holder VARCHAR(255) NOT NULL,
    phone_number VARCHAR(20),
    account_type VARCHAR(50) DEFAULT 'current',
    balance DECIMAL(15, 2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'GBP',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    transaction_type VARCHAR(50),
    category VARCHAR(50) DEFAULT 'other',
    amount DECIMAL(15, 2),
    description TEXT,
    reference VARCHAR(100),
    balance_after DECIMAL(15, 2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) UNIQUE,
    last_active TIMESTAMP DEFAULT NOW(),
    total_messages INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Registered customers — looked up by WhatsApp phone number (the "@lid") to
-- decide whether to greet with the menu or start the registration workflow.
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    aadhaar_number VARCHAR(12) UNIQUE NOT NULL,
    pan_number VARCHAR(10) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Cheque deposit requests created by the cheque workflow. request_id is the
-- unique reference the customer is given to check status later.
CREATE TABLE IF NOT EXISTS cheque_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(20) UNIQUE NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    bank_name VARCHAR(255),
    branch VARCHAR(255),
    payee VARCHAR(255),
    amount_in_figures VARCHAR(50),
    amount_in_words VARCHAR(255),
    cheque_number VARCHAR(50),
    signatory VARCHAR(255),
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS loan_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(20) UNIQUE NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    loan_type VARCHAR(50) NOT NULL,
    details JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kyc_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(20) UNIQUE NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    details JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Seed data — 3 test accounts
INSERT INTO accounts (account_number, account_holder, phone_number, account_type, balance, currency) VALUES
('GB12HSBC00010001234567', 'John Smith', '441111111111', 'current', 2543.67, 'GBP'),
('GB12HSBC00010007654321', 'Sarah Johnson', '911111111111', 'savings', 15750.00, 'GBP'),
('GB12HSBC00010009876543', 'Michael Brown', '441111111111', 'current', 892.34, 'GBP');

-- Registered customers — John Smith and Sarah Johnson are registered so the
-- greeting/menu flow has real matches. Michael Brown is deliberately left
-- unregistered so the onboarding flow has a real number to test against.
INSERT INTO customers (phone_number, full_name, aadhaar_number, pan_number) VALUES
('441111111111', 'John Smith', '234567890123', 'ABCDE1234F'),
('441111111111', 'Sarah Johnson', '345678901234', 'BXYZP5678K');

-- Transactions for John Smith (account_id=1) — 3 months, categorized,
-- balance_after runs consistently up to the seeded account balance (2543.67).
INSERT INTO transactions (account_id, transaction_type, category, amount, description, reference, balance_after, created_at) VALUES
(1, 'credit', 'salary',        2500.00, 'Salary Payment - ABC Ltd', 'SAL-M3-001', 2854.66, NOW() - INTERVAL '88 days'),
(1, 'debit',  'rent',          1200.00, 'Rent Payment',             'SO-RENT-001', 1654.66, NOW() - INTERVAL '85 days'),
(1, 'debit',  'groceries',       68.40, 'Tesco Superstore',         'POS-TES-001', 1586.26, NOW() - INTERVAL '80 days'),
(1, 'debit',  'bills',           89.50, 'British Gas',              'DD-GAS-001',  1496.76, NOW() - INTERVAL '75 days'),
(1, 'debit',  'transport',       45.00, 'TFL Travel',               'POS-TFL-001', 1451.76, NOW() - INTERVAL '70 days'),
(1, 'credit', 'salary',        2500.00, 'Salary Payment - ABC Ltd', 'SAL-M2-001',  3951.76, NOW() - INTERVAL '58 days'),
(1, 'debit',  'rent',          1200.00, 'Rent Payment',             'SO-RENT-002', 2751.76, NOW() - INTERVAL '55 days'),
(1, 'debit',  'groceries',       72.10, 'Sainsbury''s',             'POS-SAI-001', 2679.66, NOW() - INTERVAL '50 days'),
(1, 'debit',  'entertainment',   15.99, 'Netflix Subscription',     'DD-NFLX-001', 2663.67, NOW() - INTERVAL '45 days'),
(1, 'debit',  'shopping',       120.00, 'Amazon Purchase',          'POS-AMZ-001', 2543.67, NOW() - INTERVAL '40 days'),
(1, 'credit', 'salary',        2500.00, 'Salary Payment - ABC Ltd', 'SAL-JULY-2026', 5043.67, NOW() - INTERVAL '28 days'),
(1, 'debit',  'rent',          1200.00, 'Rent Payment',             'SO-RENT-003', 3843.67, NOW() - INTERVAL '25 days'),
(1, 'debit',  'bills',           89.50, 'British Gas',              'DD-GAS-002',  3754.17, NOW() - INTERVAL '18 days'),
(1, 'debit',  'groceries',       45.99, 'Tesco Superstore',         'POS-TES-002', 3708.18, NOW() - INTERVAL '10 days'),
(1, 'debit',  'other',        1164.51, 'HMRC Tax Payment',         'REF-HMRC-001', 2543.67, NOW() - INTERVAL '3 days');

-- Transactions for Sarah Johnson (account_id=2) — 3 months, ending at 15750.00
INSERT INTO transactions (account_id, transaction_type, category, amount, description, reference, balance_after, created_at) VALUES
(2, 'credit', 'transfer', 500.00, 'Transfer from Current Account', 'TRF-M3-001', 12500.00, NOW() - INTERVAL '87 days'),
(2, 'credit', 'interest', 250.00, 'Interest Payment',              'INT-M3-001', 12750.00, NOW() - INTERVAL '80 days'),
(2, 'debit',  'transfer', 200.00, 'Transfer to Current Account',   'TRF-M3-002', 12550.00, NOW() - INTERVAL '74 days'),
(2, 'credit', 'bonus',   1000.00, 'Bonus Payment',                 'BON-M3-001', 13550.00, NOW() - INTERVAL '65 days'),
(2, 'debit',  'isa',      300.00, 'ISA Transfer',                  'ISA-M3-001', 13250.00, NOW() - INTERVAL '60 days'),
(2, 'credit', 'transfer', 500.00, 'Transfer from Current Account', 'TRF-M2-001', 13750.00, NOW() - INTERVAL '55 days'),
(2, 'credit', 'interest', 260.00, 'Interest Payment',              'INT-M2-001', 14010.00, NOW() - INTERVAL '50 days'),
(2, 'debit',  'transfer', 150.00, 'Transfer to Current Account',   'TRF-M2-002', 13860.00, NOW() - INTERVAL '42 days'),
(2, 'credit', 'other',    800.00, 'Dividend Payment',              'DIV-M2-001', 14660.00, NOW() - INTERVAL '35 days'),
(2, 'debit',  'isa',      250.00, 'ISA Transfer',                  'ISA-M2-001', 14410.00, NOW() - INTERVAL '30 days'),
(2, 'credit', 'transfer', 500.00, 'Transfer from Current Account', 'TRF-M1-001', 14910.00, NOW() - INTERVAL '24 days'),
(2, 'credit', 'interest', 275.00, 'Interest Payment',              'INT-JULY-2026', 15185.00, NOW() - INTERVAL '18 days'),
(2, 'debit',  'transfer', 100.00, 'Transfer to Current Account',   'TRF-M1-002', 15085.00, NOW() - INTERVAL '12 days'),
(2, 'credit', 'bonus',    900.00, 'Bonus Payment',                 'BON-M1-001', 15985.00, NOW() - INTERVAL '6 days'),
(2, 'debit',  'isa',      235.00, 'ISA Transfer',                  'ISA-M1-001', 15750.00, NOW() - INTERVAL '2 days');

-- Transactions for Michael Brown (account_id=3) — 3 months, ending at 892.34
INSERT INTO transactions (account_id, transaction_type, category, amount, description, reference, balance_after, created_at) VALUES
(3, 'credit', 'salary',        1800.00, 'Salary Payment',       'SAL-M3-001',  1800.00, NOW() - INTERVAL '87 days'),
(3, 'debit',  'rent',           750.00, 'Rent Payment',         'SO-RENT-M3',  1050.00, NOW() - INTERVAL '84 days'),
(3, 'debit',  'entertainment',   65.00, 'Sky TV',               'DD-SKY-001',   985.00, NOW() - INTERVAL '78 days'),
(3, 'debit',  'bills',           45.00, 'Council Tax',          'DD-CT-001',    940.00, NOW() - INTERVAL '70 days'),
(3, 'debit',  'other',          940.00, 'Credit Card Payment',  'CC-PAY-001',     0.00, NOW() - INTERVAL '62 days'),
(3, 'credit', 'salary',        1800.00, 'Salary Payment',       'SAL-M2-001',  1800.00, NOW() - INTERVAL '57 days'),
(3, 'debit',  'rent',           750.00, 'Rent Payment',         'SO-RENT-M2',  1050.00, NOW() - INTERVAL '54 days'),
(3, 'debit',  'groceries',       38.50, 'Tesco Express',        'POS-TES-001', 1011.50, NOW() - INTERVAL '48 days'),
(3, 'debit',  'entertainment',   65.00, 'Sky TV',               'DD-SKY-002',   946.50, NOW() - INTERVAL '40 days'),
(3, 'debit',  'bills',           45.00, 'Council Tax',          'DD-CT-002',    901.50, NOW() - INTERVAL '32 days'),
(3, 'credit', 'salary',        1800.00, 'Salary Payment',       'SAL-JULY-2026', 2701.50, NOW() - INTERVAL '27 days'),
(3, 'debit',  'rent',           750.00, 'Rent Payment',         'SO-RENT-002', 1951.50, NOW() - INTERVAL '24 days'),
(3, 'debit',  'entertainment',   65.00, 'Sky TV',               'DD-SKY-003',  1886.50, NOW() - INTERVAL '16 days'),
(3, 'debit',  'bills',           45.00, 'Council Tax',          'DD-CT-003',   1841.50, NOW() - INTERVAL '8 days'),
(3, 'debit',  'other',          940.00, 'Credit Card Payment',  'CC-PAY-002',   892.34, NOW() - INTERVAL '2 days');

-- Sample cheque deposit requests, in different statuses, tied to John Smith
INSERT INTO cheque_requests (request_id, phone_number, bank_name, branch, payee, amount_in_figures, amount_in_words, cheque_number, signatory, status, created_at) VALUES
('CHQ-A1B2C3D4', '441111111111', 'HSBC', 'London Central', 'John Smith', '500.00', 'Five Hundred Pounds Only', '000123', 'A. Patel', 'COMPLETED', NOW() - INTERVAL '10 days'),
('CHQ-E5F6G7H8', '441111111111', 'HSBC', 'London Central', 'John Smith', '1200.00', 'One Thousand Two Hundred Pounds Only', '000124', 'A. Patel', 'PENDING', NOW() - INTERVAL '1 day'),
('CHQ-J9K1L2M3', '441111111111', 'Barclays', 'Manchester', 'Sarah Johnson', '75.00', 'Seventy Five Pounds Only', '000045', 'R. Khan', 'REJECTED', NOW() - INTERVAL '5 days');
