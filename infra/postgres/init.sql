-- Finacle Banking WhatsApp Assistant — Database Schema

CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    account_number VARCHAR(30) UNIQUE NOT NULL,
    account_holder VARCHAR(255) NOT NULL,
    phone_number VARCHAR(20),
    account_type VARCHAR(50) DEFAULT 'current',
    balance DECIMAL(15, 2) DEFAULT 20000.00,
    currency VARCHAR(3) DEFAULT 'INR',
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
    date_of_birth VARCHAR(50),
    guardian_name VARCHAR(255),
    address TEXT,
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
    date_written VARCHAR(50),
    drawer_name VARCHAR(255),
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

-- Saved beneficiaries for the money-transfer workflow. Owned by the
-- customer's phone number; (phone_number, account_number) is unique so
-- re-adding the same beneficiary updates their name instead of duplicating.
CREATE TABLE IF NOT EXISTS beneficiaries (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) NOT NULL,
    beneficiary_name VARCHAR(255) NOT NULL,
    account_number VARCHAR(30) NOT NULL,
    bank_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (phone_number, account_number)
);

-- Money transfers. Created as INITIATED the moment the customer confirms —
-- there is no OTP/SMS step. status is updated later (e.g. by bank staff, via
-- the debug endpoint) to COMPLETED or FAILED once the transfer actually
-- settles on the bank side.
CREATE TABLE IF NOT EXISTS transfers (
    id SERIAL PRIMARY KEY,
    reference VARCHAR(20) UNIQUE NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    source_account VARCHAR(30) NOT NULL,
    beneficiary_name VARCHAR(255) NOT NULL,
    beneficiary_account VARCHAR(30) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    status VARCHAR(20) DEFAULT 'INITIATED',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Published loan product terms — the bank's own rate card. Static reference
-- data, not a customer's application (see loan_requests for that). Exists so
-- the assistant can answer "what's the interest rate on a personal loan?"
-- with a real, tool-provided figure instead of either inventing one or
-- refusing to answer — see app/agent/tools.py::tool_get_loan_product_info().
-- Rates are illustrative representative APRs for this demo bank, not live
-- market rates.
CREATE TABLE IF NOT EXISTS loan_products (
    id SERIAL PRIMARY KEY,
    loan_type VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    interest_rate_min DECIMAL(5, 2) NOT NULL,
    interest_rate_max DECIMAL(5, 2) NOT NULL,
    min_amount DECIMAL(15, 2) NOT NULL,
    max_amount DECIMAL(15, 2) NOT NULL,
    min_tenure_months INTEGER NOT NULL,
    max_tenure_months INTEGER NOT NULL,
    processing_fee_percent DECIMAL(4, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'INR',
    notes TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO loan_products (loan_type, display_name, interest_rate_min, interest_rate_max, min_amount, max_amount, min_tenure_months, max_tenure_months, processing_fee_percent, currency, notes) VALUES
('personal',  'Personal Loan',  10.50, 15.00,    10000.00,  1000000.00, 2, 60, 1.50, 'INR', 'Unsecured. Final rate depends on credit profile and income.'),
('home',      'Home Loan',       6.75,  9.25,   100000.00, 4000000.00, 12, 78, 0.50, 'INR', 'Secured against the property. Final rate depends on loan-to-value and credit profile.'),
('vehicle',   'Vehicle Loan',    8.50, 12.00,    20000.00, 1500000.00, 2, 60, 1.00, 'INR', 'Secured against the vehicle. Final rate depends on vehicle age and credit profile.'),
('education', 'Education Loan',  7.00, 11.00,    10000.00, 2000000.00, 2, 78, 0.75, 'INR', 'Repayment can often be deferred until after course completion — ask about moratorium terms.');

-- Seed data — 3 test accounts
INSERT INTO accounts (account_number, account_holder, phone_number, account_type, balance, currency) VALUES
('FNCL000000000001', 'Aditya',  '911111111111', 'savings', 20000.00, 'INR'),
('FNCL000000000002', 'Veena',   '911111111111', 'savings', 20000.00, 'INR'),
('FNCL000000000003', 'Lavanya', '911111111111', 'savings', 20000.00, 'INR');


-- Registered customers — John Smith and Sarah Johnson are registered so the
-- greeting/menu flow has real matches. Michael Brown is deliberately left
-- unregistered so the onboarding flow has a real number to test against.
-- Phone numbers here must match the accounts row above for each person —
-- the registration gate and get_accounts_by_phone() both key off this
-- number, so a mismatch makes a "registered" customer look unregistered.
INSERT INTO customers (id, phone_number, full_name, aadhaar_number, pan_number, date_of_birth, guardian_name, address, status, created_at) VALUES (1, '911111111111', 'Aditya', '111122223333', 'ADITY1234A', '1995-05-15', 'Ramesh Kumar', 'Visakhapatnam, Andhra Pradesh, India', 'active', NOW()), (2, '911111111111', 'Veena', '222233334444', 'VEENA1234B', '1996-08-20', 'Suresh Kumar', 'Visakhapatnam, Andhra Pradesh, India', 'active', NOW()), (3, '911111111111', 'Lavanya', '333344445555', 'LAVAN1234C', '1997-11-10', 'Ravi Kumar', 'Visakhapatnam, Andhra Pradesh, India', 'active', NOW());

-- Aditya: 15 transactions, ending balance = 20,000.00
INSERT INTO transactions (account_id, transaction_type, category, amount, description, reference, balance_after, created_at) VALUES
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'salary',      5000.00, 'Salary Credit',              'ADI-SAL-001', 25000.00, NOW() - INTERVAL '90 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'rent',        3000.00, 'House Rent',                  'ADI-REN-001', 22000.00, NOW() - INTERVAL '85 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'groceries',    850.00, 'Grocery Shopping',            'ADI-GRC-001', 21150.00, NOW() - INTERVAL '80 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     1500.00, 'Transfer Received',           'ADI-TRF-001', 22650.00, NOW() - INTERVAL '75 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'bills',         650.00, 'Utility Bill Payment',        'ADI-BIL-001', 22000.00, NOW() - INTERVAL '70 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'shopping',      900.00, 'Online Shopping',             'ADI-SHP-001', 21100.00, NOW() - INTERVAL '65 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'salary',       5000.00, 'Salary Credit',              'ADI-SAL-002', 26100.00, NOW() - INTERVAL '60 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'rent',        3000.00, 'House Rent',                  'ADI-REN-002', 23100.00, NOW() - INTERVAL '55 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'transport',    700.00, 'Transport Expenses',          'ADI-TRN-001', 22400.00, NOW() - INTERVAL '50 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'groceries',   1100.00, 'Grocery Shopping',            'ADI-GRC-002', 21300.00, NOW() - INTERVAL '40 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     800.00, 'Transfer Received',           'ADI-TRF-002', 22100.00, NOW() - INTERVAL '30 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'bills',         600.00, 'Utility Bill Payment',        'ADI-BIL-002', 21500.00, NOW() - INTERVAL '25 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'shopping',      750.00, 'Online Shopping',             'ADI-SHP-002', 20750.00, NOW() - INTERVAL '18 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     1000.00, 'Transfer Received',           'ADI-TRF-003', 21750.00, NOW() - INTERVAL '10 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'other',        1750.00, 'Miscellaneous Payment',       'ADI-OTH-001', 20000.00, NOW() - INTERVAL '2 days');

-- Veena: 15 transactions, ending balance = 20,000.00
INSERT INTO transactions (account_id, transaction_type, category, amount, description, reference, balance_after, created_at) VALUES
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'salary',      6000.00, 'Salary Credit',              'VEE-SAL-001', 26000.00, NOW() - INTERVAL '90 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'rent',        3500.00, 'House Rent',                  'VEE-REN-001', 22500.00, NOW() - INTERVAL '85 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'groceries',   1000.00, 'Grocery Shopping',            'VEE-GRC-001', 21500.00, NOW() - INTERVAL '80 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     2000.00, 'Transfer Received',           'VEE-TRF-001', 23500.00, NOW() - INTERVAL '75 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'bills',         800.00, 'Utility Bill Payment',        'VEE-BIL-001', 22700.00, NOW() - INTERVAL '70 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'shopping',     1200.00, 'Online Shopping',             'VEE-SHP-001', 21500.00, NOW() - INTERVAL '65 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'salary',      6000.00, 'Salary Credit',              'VEE-SAL-002', 27500.00, NOW() - INTERVAL '60 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'rent',        3500.00, 'House Rent',                  'VEE-REN-002', 24000.00, NOW() - INTERVAL '55 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'transport',    900.00, 'Transport Expenses',          'VEE-TRN-001', 23100.00, NOW() - INTERVAL '50 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'groceries',   1300.00, 'Grocery Shopping',            'VEE-GRC-002', 21800.00, NOW() - INTERVAL '40 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     700.00, 'Transfer Received',           'VEE-TRF-002', 22500.00, NOW() - INTERVAL '30 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'bills',         500.00, 'Utility Bill Payment',        'VEE-BIL-002', 22000.00, NOW() - INTERVAL '25 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'shopping',     1000.00, 'Online Shopping',             'VEE-SHP-002', 21000.00, NOW() - INTERVAL '18 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     1500.00, 'Transfer Received',           'VEE-TRF-003', 22500.00, NOW() - INTERVAL '10 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'other',        2500.00, 'Miscellaneous Payment',       'VEE-OTH-001', 20000.00, NOW() - INTERVAL '2 days');

-- Lavanya: 15 transactions, ending balance = 20,000.00
INSERT INTO transactions (account_id, transaction_type, category, amount, description, reference, balance_after, created_at) VALUES
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'salary',      4500.00, 'Salary Credit',              'LAV-SAL-001', 24500.00, NOW() - INTERVAL '90 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'rent',        2800.00, 'House Rent',                  'LAV-REN-001', 21700.00, NOW() - INTERVAL '85 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'groceries',    900.00, 'Grocery Shopping',            'LAV-GRC-001', 20800.00, NOW() - INTERVAL '80 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     1800.00, 'Transfer Received',           'LAV-TRF-001', 22600.00, NOW() - INTERVAL '75 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'bills',         700.00, 'Utility Bill Payment',        'LAV-BIL-001', 21900.00, NOW() - INTERVAL '70 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'shopping',      600.00, 'Online Shopping',             'LAV-SHP-001', 21300.00, NOW() - INTERVAL '65 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'salary',      4500.00, 'Salary Credit',              'LAV-SAL-002', 25800.00, NOW() - INTERVAL '60 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'rent',        2800.00, 'House Rent',                  'LAV-REN-002', 23000.00, NOW() - INTERVAL '55 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'transport',    650.00, 'Transport Expenses',          'LAV-TRN-001', 22350.00, NOW() - INTERVAL '50 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'groceries',   1050.00, 'Grocery Shopping',            'LAV-GRC-002', 21300.00, NOW() - INTERVAL '40 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     900.00, 'Transfer Received',           'LAV-TRF-002', 22200.00, NOW() - INTERVAL '30 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'bills',         550.00, 'Utility Bill Payment',        'LAV-BIL-002', 21650.00, NOW() - INTERVAL '25 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'shopping',      850.00, 'Online Shopping',             'LAV-SHP-002', 20800.00, NOW() - INTERVAL '18 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'credit', 'transfer',     1200.00, 'Transfer Received',           'LAV-TRF-003', 22000.00, NOW() - INTERVAL '10 days'),
((SELECT id FROM accounts WHERE phone_number = '911111111111'), 'debit',  'other',        2000.00, 'Miscellaneous Payment',       'LAV-OTH-001', 20000.00, NOW() - INTERVAL '2 days');

-- Sample cheque deposit requests, in different statuses, tied to John Smith
INSERT INTO cheque_requests (request_id, phone_number, bank_name, branch, payee, amount_in_figures, amount_in_words, cheque_number, signatory, status, created_at) VALUES
('CHQ-A1B2C3D4', '447818658034', 'Finacle Banking', 'London Central', 'John Smith', '500.00', 'Five Hundred Pounds Only', '000123', 'A. Patel', 'COMPLETED', NOW() - INTERVAL '10 days'),
('CHQ-E5F6G7H8', '447818658034', 'Finacle Banking', 'London Central', 'John Smith', '1200.00', 'One Thousand Two Hundred Pounds Only', '000124', 'A. Patel', 'PENDING', NOW() - INTERVAL '1 day'),
('CHQ-J9K1L2M3', '911111111111', 'Barclays', 'Manchester', 'Sarah Johnson', '75.00', 'Seventy Five Pounds Only', '000045', 'R. Khan', 'REJECTED', NOW() - INTERVAL '5 days');

-- Saved beneficiaries so the transfer workflow has real data to list
INSERT INTO beneficiaries (phone_number, beneficiary_name, account_number, bank_name) VALUES
('447818658034', 'Priya Sharma', 'GB29FNCL60161331926819', 'Finacle Banking'),
('447818658034', 'Alex Morgan', 'GB77FNCL29001847502211', 'Finacle Banking'),
('911111111111', 'Rahul Verma', 'GB14FNCL74208891736642', 'Finacle Banking'),
('911111111111', 'Emma Wilson', 'GB05FNCL13590027461938', 'Finacle Banking');
