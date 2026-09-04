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

-- Seed data — accounts (synced from live database snapshot, 2026-09-04)
INSERT INTO accounts (account_number, account_holder, phone_number, account_type, balance, currency) VALUES
('FNCL000000000001',     'Aditya',       '919579145784', 'savings', 20000.00,  'INR'),
('FNCL000000000002',     'Veena',        '917012821406', 'savings', 10000.00,  'INR'),
('FNCL000000000003',     'Lavanya',      '919384148239', 'savings', 10949.00,  'INR'),
('FNCL000000000004',     'Jeena',        '917829918363', 'savings', 13079.00,  'INR'),
('FNCL000000000005',     'Priya P',      '918956140132', 'savings', 20000.00,  'INR'),
('INFIAAE8F0009B7A3CB0', 'WILLAM BROWN', '917331122909', 'savings', 15950.00,  'INR'),
('FNCLFAIRACC00004',     'Ramprasath G', '919843222825', 'savings', 45000.00,  'INR'),
('INFI392CF333EB2B2B1C', 'WILLAM BROWN', '917331122909', 'current', 18750.00,  'INR'),
('INFIBCAFC70551D755ED', 'Jeena',        '917829918363', 'current', 20000.00,  'INR'),
('FNCL000000000006',     'Thanu',        '916384354499', 'savings', 29000.00,  'INR'),
('FNCLFAIRACC00005',     'Ramprasath G', '919843222825', 'current', 150000.00, 'INR'),
('INFI5AC453AA27E1463F', 'Jeena',        '917829918363', 'salary',  20000.00,  'INR');


-- Registered customers — synced from live database snapshot, 2026-09-04.
-- Phone numbers here must match the accounts rows above for each person —
-- the registration gate and get_accounts_by_phone() both key off this
-- number, so a mismatch makes a "registered" customer look unregistered.
INSERT INTO customers (id, phone_number, full_name, aadhaar_number, pan_number, date_of_birth, guardian_name, address, status, created_at) VALUES (1, '919579145784', 'Aditya', '111122223333', 'ADITY1234A', '1995-05-15', 'Ramesh Kumar', 'Visakhapatnam, Andhra Pradesh, India', 'active', NOW()), (2, '917012821406', 'Veena', '222233334444', 'VEENA1234B', '1996-08-20', 'Suresh Kumar', 'Visakhapatnam, Andhra Pradesh, India', 'active', NOW()), (3, '919384148239', 'Lavanya', '333344445555', 'LAVAN1234C', '1997-11-10', 'Ravi Kumar', 'Visakhapatnam, Andhra Pradesh, India', 'active', NOW()), (4, '917829918363', 'Jeena', '123456789012', 'ABCDE1234F', '1994-03-12', 'Suresh Prasad', 'Hyderabad, Telangana', 'active', NOW()), (5, '918956140132', 'Priya P', '555566667777', 'PRIYA1234P', '1995-07-25', 'Suresh Kumar', 'Bengaluru, Karnataka', 'active', NOW()), (6, '917331122909', 'WILLAM BROWN', '582791436611', 'BYDPB4818H', '18/05/1978', 'Ramesh Babu', 'H.No. 4B/2, 3rd Floor, Tulsi Apartment, Adyar Main Road, Chennai, Tamil Nadu - 600020', 'active', NOW()), (7, '919843222825', 'Ramprasath G', '123456789013', 'ABCDE1234G', '15/06/1985', 'Prasad Babu', '123, Shanti Nagar, Near Park, Lucknow, Uttar Pradesh - 226001', 'active', NOW()), (8, '916384354499', 'Thanu', '123456789014', 'ABCDE1234T', '15/06/1999', 'Krishna', 'Bangalore', 'active', NOW());

-- Transactions — full ledger synced from live database snapshot, 2026-09-04
-- (224 rows, account_id refers to the accounts rows above in insertion order:
-- 1 Aditya, 2 Veena, 3 Lavanya, 4 Jeena/savings, 5 Priya P, 6 WILLAM BROWN/savings,
-- 7 Ramprasath G/savings, 8 WILLAM BROWN/current, 9 Jeena/current, 10 Thanu,
-- 11 Ramprasath G/current, 12 Jeena/salary)
INSERT INTO transactions (id, account_id, transaction_type, category, amount, description, reference, balance_after, created_at) VALUES
(1, 1, 'credit', 'salary', 5000.00, 'Salary Credit', 'ADI-SAL-001', 25000.00, '2026-05-20 08:51:47.934207'),
(2, 1, 'debit', 'rent', 3000.00, 'House Rent', 'ADI-REN-001', 22000.00, '2026-05-25 08:51:47.934207'),
(3, 1, 'debit', 'groceries', 850.00, 'Grocery Shopping', 'ADI-GRC-001', 21150.00, '2026-05-30 08:51:47.934207'),
(4, 1, 'credit', 'transfer', 1500.00, 'Transfer Received', 'ADI-TRF-001', 22650.00, '2026-06-04 08:51:47.934207'),
(5, 1, 'debit', 'bills', 650.00, 'Utility Bill Payment', 'ADI-BIL-001', 22000.00, '2026-06-09 08:51:47.934207'),
(6, 1, 'debit', 'shopping', 900.00, 'Online Shopping', 'ADI-SHP-001', 21100.00, '2026-06-14 08:51:47.934207'),
(7, 1, 'credit', 'salary', 5000.00, 'Salary Credit', 'ADI-SAL-002', 26100.00, '2026-06-19 08:51:47.934207'),
(8, 1, 'debit', 'rent', 3000.00, 'House Rent', 'ADI-REN-002', 23100.00, '2026-06-24 08:51:47.934207'),
(9, 1, 'debit', 'transport', 700.00, 'Transport Expenses', 'ADI-TRN-001', 22400.00, '2026-06-29 08:51:47.934207'),
(10, 1, 'debit', 'groceries', 1100.00, 'Grocery Shopping', 'ADI-GRC-002', 21300.00, '2026-07-09 08:51:47.934207'),
(11, 1, 'credit', 'transfer', 800.00, 'Transfer Received', 'ADI-TRF-002', 22100.00, '2026-07-19 08:51:47.934207'),
(12, 1, 'debit', 'bills', 600.00, 'Utility Bill Payment', 'ADI-BIL-002', 21500.00, '2026-07-24 08:51:47.934207'),
(13, 1, 'debit', 'shopping', 750.00, 'Online Shopping', 'ADI-SHP-002', 20750.00, '2026-07-31 08:51:47.934207'),
(14, 1, 'credit', 'transfer', 1000.00, 'Transfer Received', 'ADI-TRF-003', 21750.00, '2026-08-08 08:51:47.934207'),
(15, 1, 'debit', 'other', 1750.00, 'Miscellaneous Payment', 'ADI-OTH-001', 20000.00, '2026-08-16 08:51:47.934207'),
(16, 2, 'credit', 'salary', 6000.00, 'Salary Credit', 'VEE-SAL-001', 26000.00, '2026-05-20 08:51:47.934207'),
(17, 2, 'debit', 'rent', 3500.00, 'House Rent', 'VEE-REN-001', 22500.00, '2026-05-25 08:51:47.934207'),
(18, 2, 'debit', 'groceries', 1000.00, 'Grocery Shopping', 'VEE-GRC-001', 21500.00, '2026-05-30 08:51:47.934207'),
(19, 2, 'credit', 'transfer', 2000.00, 'Transfer Received', 'VEE-TRF-001', 23500.00, '2026-06-04 08:51:47.934207'),
(20, 2, 'debit', 'bills', 800.00, 'Utility Bill Payment', 'VEE-BIL-001', 22700.00, '2026-06-09 08:51:47.934207'),
(21, 2, 'debit', 'shopping', 1200.00, 'Online Shopping', 'VEE-SHP-001', 21500.00, '2026-06-14 08:51:47.934207'),
(22, 2, 'credit', 'salary', 6000.00, 'Salary Credit', 'VEE-SAL-002', 27500.00, '2026-06-19 08:51:47.934207'),
(23, 2, 'debit', 'rent', 3500.00, 'House Rent', 'VEE-REN-002', 24000.00, '2026-06-24 08:51:47.934207'),
(24, 2, 'debit', 'transport', 900.00, 'Transport Expenses', 'VEE-TRN-001', 23100.00, '2026-06-29 08:51:47.934207'),
(25, 2, 'debit', 'groceries', 1300.00, 'Grocery Shopping', 'VEE-GRC-002', 21800.00, '2026-07-09 08:51:47.934207'),
(26, 2, 'credit', 'transfer', 700.00, 'Transfer Received', 'VEE-TRF-002', 22500.00, '2026-07-19 08:51:47.934207'),
(27, 2, 'debit', 'bills', 500.00, 'Utility Bill Payment', 'VEE-BIL-002', 22000.00, '2026-07-24 08:51:47.934207'),
(28, 2, 'debit', 'shopping', 1000.00, 'Online Shopping', 'VEE-SHP-002', 21000.00, '2026-07-31 08:51:47.934207'),
(29, 2, 'credit', 'transfer', 1500.00, 'Transfer Received', 'VEE-TRF-003', 22500.00, '2026-08-08 08:51:47.934207'),
(30, 2, 'debit', 'other', 2500.00, 'Miscellaneous Payment', 'VEE-OTH-001', 20000.00, '2026-08-16 08:51:47.934207'),
(31, 3, 'credit', 'salary', 4500.00, 'Salary Credit', 'LAV-SAL-001', 24500.00, '2026-05-20 08:51:47.934207'),
(32, 3, 'debit', 'rent', 2800.00, 'House Rent', 'LAV-REN-001', 21700.00, '2026-05-25 08:51:47.934207'),
(33, 3, 'debit', 'groceries', 900.00, 'Grocery Shopping', 'LAV-GRC-001', 20800.00, '2026-05-30 08:51:47.934207'),
(34, 3, 'credit', 'transfer', 1800.00, 'Transfer Received', 'LAV-TRF-001', 22600.00, '2026-06-04 08:51:47.934207'),
(35, 3, 'debit', 'bills', 700.00, 'Utility Bill Payment', 'LAV-BIL-001', 21900.00, '2026-06-09 08:51:47.934207'),
(36, 3, 'debit', 'shopping', 600.00, 'Online Shopping', 'LAV-SHP-001', 21300.00, '2026-06-14 08:51:47.934207'),
(37, 3, 'credit', 'salary', 4500.00, 'Salary Credit', 'LAV-SAL-002', 25800.00, '2026-06-19 08:51:47.934207'),
(38, 3, 'debit', 'rent', 2800.00, 'House Rent', 'LAV-REN-002', 23000.00, '2026-06-24 08:51:47.934207'),
(39, 3, 'debit', 'transport', 650.00, 'Transport Expenses', 'LAV-TRN-001', 22350.00, '2026-06-29 08:51:47.934207'),
(40, 3, 'debit', 'groceries', 1050.00, 'Grocery Shopping', 'LAV-GRC-002', 21300.00, '2026-07-09 08:51:47.934207'),
(41, 3, 'credit', 'transfer', 900.00, 'Transfer Received', 'LAV-TRF-002', 22200.00, '2026-07-19 08:51:47.934207'),
(42, 3, 'debit', 'bills', 550.00, 'Utility Bill Payment', 'LAV-BIL-002', 21650.00, '2026-07-24 08:51:47.934207'),
(43, 3, 'debit', 'shopping', 850.00, 'Online Shopping', 'LAV-SHP-002', 20800.00, '2026-07-31 08:51:47.934207'),
(44, 3, 'credit', 'transfer', 1200.00, 'Transfer Received', 'LAV-TRF-003', 22000.00, '2026-08-08 08:51:47.934207'),
(45, 3, 'debit', 'other', 2000.00, 'Miscellaneous Payment', 'LAV-OTH-001', 20000.00, '2026-08-16 08:51:47.934207'),
(46, 2, 'debit', 'transfer', 10000.00, 'Transfer to My Landlord If Balance Is More Than', 'TRF-C9D0341A', 10000.00, '2026-08-18 09:22:26.740171'),
(47, 4, 'credit', 'salary', 6000.00, 'Salary Credit', 'JEE-SAL-001', 36000.00, '2026-05-21 05:52:37.922914'),
(48, 4, 'debit', 'rent', 4000.00, 'House Rent', 'JEE-REN-001', 32000.00, '2026-05-26 05:52:37.922914'),
(49, 4, 'debit', 'groceries', 1200.00, 'Supermarket', 'JEE-GRC-001', 30800.00, '2026-05-31 05:52:37.922914'),
(50, 4, 'credit', 'bonus', 2500.00, 'Annual Bonus', 'JEE-BON-001', 33300.00, '2026-06-05 05:52:37.922914'),
(51, 4, 'debit', 'bills', 900.00, 'Electricity Bill', 'JEE-BIL-001', 32400.00, '2026-06-10 05:52:37.922914'),
(52, 4, 'debit', 'shopping', 1500.00, 'Online Shopping', 'JEE-SHP-001', 30900.00, '2026-06-15 05:52:37.922914'),
(53, 4, 'credit', 'salary', 6000.00, 'Salary Credit', 'JEE-SAL-002', 36900.00, '2026-06-20 05:52:37.922914'),
(54, 4, 'debit', 'rent', 4000.00, 'House Rent', 'JEE-REN-002', 32900.00, '2026-06-25 05:52:37.922914'),
(55, 4, 'debit', 'transport', 700.00, 'Cab Ride', 'JEE-TRN-001', 32200.00, '2026-06-30 05:52:37.922914'),
(56, 4, 'debit', 'groceries', 1000.00, 'Supermarket', 'JEE-GRC-002', 31200.00, '2026-07-10 05:52:37.922914'),
(57, 4, 'credit', 'transfer', 1200.00, 'Transfer Received', 'JEE-TRF-001', 32400.00, '2026-07-20 05:52:37.922914'),
(58, 4, 'debit', 'bills', 800.00, 'Internet Bill', 'JEE-BIL-002', 31600.00, '2026-07-25 05:52:37.922914'),
(59, 4, 'debit', 'shopping', 900.00, 'Clothing Store', 'JEE-SHP-002', 30700.00, '2026-08-01 05:52:37.922914'),
(60, 4, 'credit', 'refund', 1000.00, 'Product Refund', 'JEE-REF-001', 31700.00, '2026-08-09 05:52:37.922914'),
(61, 4, 'debit', 'other', 1700.00, 'Miscellaneous Payment', 'JEE-OTH-001', 30000.00, '2026-08-17 05:52:37.922914'),
(62, 5, 'credit', 'salary', 5000.00, 'Salary Credit', 'PRI-SAL-001', 25000.00, '2026-05-21 05:52:37.922914'),
(63, 5, 'debit', 'rent', 3000.00, 'House Rent', 'PRI-REN-001', 22000.00, '2026-05-26 05:52:37.922914'),
(64, 5, 'debit', 'groceries', 900.00, 'Supermarket', 'PRI-GRC-001', 21100.00, '2026-05-31 05:52:37.922914'),
(65, 5, 'credit', 'bonus', 2000.00, 'Annual Bonus', 'PRI-BON-001', 23100.00, '2026-06-05 05:52:37.922914'),
(66, 5, 'debit', 'bills', 700.00, 'Electricity Bill', 'PRI-BIL-001', 22400.00, '2026-06-10 05:52:37.922914'),
(67, 5, 'debit', 'shopping', 1200.00, 'Online Shopping', 'PRI-SHP-001', 21200.00, '2026-06-15 05:52:37.922914'),
(68, 5, 'credit', 'salary', 5000.00, 'Salary Credit', 'PRI-SAL-002', 26200.00, '2026-06-20 05:52:37.922914'),
(69, 5, 'debit', 'rent', 3000.00, 'House Rent', 'PRI-REN-002', 23200.00, '2026-06-25 05:52:37.922914'),
(70, 5, 'debit', 'transport', 600.00, 'Cab Ride', 'PRI-TRN-001', 22600.00, '2026-06-30 05:52:37.922914'),
(71, 5, 'debit', 'groceries', 1000.00, 'Supermarket', 'PRI-GRC-002', 21600.00, '2026-07-10 05:52:37.922914'),
(72, 5, 'credit', 'transfer', 800.00, 'Transfer Received', 'PRI-TRF-001', 22400.00, '2026-07-20 05:52:37.922914'),
(73, 5, 'debit', 'bills', 600.00, 'Internet Bill', 'PRI-BIL-002', 21800.00, '2026-07-25 05:52:37.922914'),
(74, 5, 'debit', 'shopping', 1000.00, 'Clothing Store', 'PRI-SHP-002', 20800.00, '2026-08-01 05:52:37.922914'),
(75, 5, 'credit', 'refund', 1200.00, 'Product Refund', 'PRI-REF-001', 22000.00, '2026-08-09 05:52:37.922914'),
(76, 5, 'debit', 'other', 2000.00, 'Miscellaneous Payment', 'PRI-OTH-001', 20000.00, '2026-08-17 05:52:37.922914'),
(77, 4, 'debit', 'transfer', 2000.00, 'Transfer to Bhavitha', 'TRF-C437D961', 28000.00, '2026-08-19 06:39:27.31804'),
(78, 6, 'credit', 'salary', 2500.00, 'Salary credit', 'INIT-001', 2500.00, '2026-08-19 06:41:02.275027'),
(79, 6, 'debit', 'rent', 1200.00, 'Rent payment', 'INIT-002', 1300.00, '2026-08-19 06:41:02.275027'),
(80, 6, 'debit', 'groceries', 300.00, 'Groceries', 'INIT-003', 1000.00, '2026-08-19 06:41:02.275027'),
(81, 6, 'credit', 'bonus', 1500.00, 'Bonus payout', 'INIT-004', 2500.00, '2026-08-19 06:41:02.275027'),
(82, 6, 'debit', 'bills', 700.00, 'Utility bill', 'INIT-005', 1800.00, '2026-08-19 06:41:02.275027'),
(83, 6, 'debit', 'transport', 450.00, 'Transport', 'INIT-006', 1350.00, '2026-08-19 06:41:02.275027'),
(84, 6, 'credit', 'salary', 2200.00, 'Salary credit', 'INIT-007', 3550.00, '2026-08-19 06:41:02.275027'),
(85, 6, 'debit', 'shopping', 600.00, 'Shopping', 'INIT-008', 2950.00, '2026-08-19 06:41:02.275027'),
(86, 6, 'debit', 'entertainment', 350.00, 'Streaming services', 'INIT-009', 2600.00, '2026-08-19 06:41:02.275027'),
(87, 6, 'credit', 'transfer', 1200.00, 'Internal transfer', 'INIT-010', 3800.00, '2026-08-19 06:41:02.275027'),
(88, 6, 'debit', 'groceries', 200.00, 'Groceries', 'INIT-011', 3600.00, '2026-08-19 06:41:02.275027'),
(89, 6, 'credit', 'bonus', 4500.00, 'Performance bonus', 'INIT-012', 8100.00, '2026-08-19 06:41:02.275027'),
(90, 6, 'debit', 'rent', 1100.00, 'Rent payment', 'INIT-013', 7000.00, '2026-08-19 06:41:02.275027'),
(91, 6, 'credit', 'transfer', 9000.00, 'Account top-up', 'INIT-014', 16000.00, '2026-08-19 06:41:02.275027'),
(92, 6, 'credit', 'other', 4000.00, 'Account opening balance', 'INIT-015', 20000.00, '2026-08-19 06:41:02.275027'),
(93, 6, 'debit', 'transfer', 2000.00, 'Transfer to Bhavitha', 'TRF-7F19CED5', 18000.00, '2026-08-19 07:59:25.320716'),
(94, 6, 'debit', 'transfer', 500.00, 'Transfer to Bhavitha', 'TRF-3B490E0A', 17500.00, '2026-08-19 08:28:28.677343'),
(95, 6, 'debit', 'transfer', 500.00, 'Transfer to Bhavitha', 'TRF-093B2FFB', 17000.00, '2026-08-19 08:52:40.01735'),
(96, 4, 'debit', 'transfer', 2000.00, 'Transfer to Bhavita', 'TRF-A6993E26', 26000.00, '2026-08-19 08:58:09.978889'),
(97, 4, 'debit', 'transfer', 1000.00, 'Transfer to Chinmay', 'TRF-B9B1E779', 25000.00, '2026-08-19 09:00:04.778384'),
(98, 7, 'debit', 'travel', 12000.00, 'Flight Trip', 'VP-TRV-001', 288000.00, '2026-05-21 14:56:07.17669'),
(99, 7, 'debit', 'travel', 8000.00, 'Hotel Stay', 'VP-TRV-002', 280000.00, '2026-05-26 14:56:07.17669'),
(100, 7, 'debit', 'travel', 6000.00, 'Cab Ride', 'VP-TRV-003', 274000.00, '2026-05-31 14:56:07.17669'),
(101, 7, 'debit', 'travel', 15000.00, 'Business Tour', 'VP-TRV-004', 259000.00, '2026-06-05 14:56:07.17669'),
(102, 7, 'debit', 'travel', 7000.00, 'Business Tour to Delhi', 'VP-TRV-005', 252000.00, '2026-06-10 14:56:07.17669'),
(103, 7, 'debit', 'shopping', 5000.00, 'Shopping', 'VP-SHP-001', 247000.00, '2026-06-15 14:56:07.17669'),
(104, 7, 'debit', 'shopping', 4000.00, 'Shopping', 'VP-SHP-002', 243000.00, '2026-06-20 14:56:07.17669'),
(105, 7, 'debit', 'shopping', 3500.00, 'Shoes Order', 'VP-SHP-003', 239500.00, '2026-06-25 14:56:07.17669'),
(106, 7, 'debit', 'shopping', 6000.00, 'Gift Items', 'VP-SHP-004', 233500.00, '2026-06-30 14:56:07.17669'),
(107, 7, 'debit', 'shopping', 4500.00, 'Office Decor', 'VP-SHP-005', 229000.00, '2026-07-05 14:56:07.17669'),
(108, 7, 'debit', 'corporate_dining', 3000.00, 'Client Lunch', 'VP-DIN-001', 226000.00, '2026-07-10 14:56:07.17669'),
(109, 7, 'debit', 'corporate_dining', 3500.00, 'Team Dinner', 'VP-DIN-002', 222500.00, '2026-07-15 14:56:07.17669'),
(110, 7, 'debit', 'corporate_dining', 2500.00, 'Partner Meal', 'VP-DIN-003', 220000.00, '2026-07-20 14:56:07.17669'),
(111, 7, 'debit', 'bills', 2000.00, 'Phone Bill', 'VP-BIL-001', 218000.00, '2026-07-25 14:56:07.17669'),
(112, 7, 'debit', 'bills', 2500.00, 'Power Bill', 'VP-BIL-002', 215500.00, '2026-07-30 14:56:07.17669'),
(113, 7, 'credit', 'salary', 400000.00, 'Exec Salary', 'VP-SAL-001', 515500.00, '2026-08-09 14:56:07.17669'),
(114, 6, 'debit', 'transfer', 50.00, 'Transfer to Bhavitha', 'TRF-4902B364', 16950.00, '2026-08-19 16:44:14.476929'),
(115, 8, 'credit', 'salary', 2500.00, 'Salary credit', 'INIT-001', 2500.00, '2026-08-19 18:23:40.48105'),
(116, 8, 'debit', 'rent', 1200.00, 'Rent payment', 'INIT-002', 1300.00, '2026-08-19 18:23:40.48105'),
(117, 8, 'debit', 'groceries', 300.00, 'Groceries', 'INIT-003', 1000.00, '2026-08-19 18:23:40.48105'),
(118, 8, 'credit', 'bonus', 1500.00, 'Bonus payout', 'INIT-004', 2500.00, '2026-08-19 18:23:40.48105'),
(119, 8, 'debit', 'bills', 700.00, 'Utility bill', 'INIT-005', 1800.00, '2026-08-19 18:23:40.48105'),
(120, 8, 'debit', 'transport', 450.00, 'Transport', 'INIT-006', 1350.00, '2026-08-19 18:23:40.48105'),
(121, 8, 'credit', 'salary', 2200.00, 'Salary credit', 'INIT-007', 3550.00, '2026-08-19 18:23:40.48105'),
(122, 8, 'debit', 'shopping', 600.00, 'Shopping', 'INIT-008', 2950.00, '2026-08-19 18:23:40.48105'),
(123, 8, 'debit', 'entertainment', 350.00, 'Streaming services', 'INIT-009', 2600.00, '2026-08-19 18:23:40.48105'),
(124, 8, 'credit', 'transfer', 1200.00, 'Internal transfer', 'INIT-010', 3800.00, '2026-08-19 18:23:40.48105'),
(125, 8, 'debit', 'groceries', 200.00, 'Groceries', 'INIT-011', 3600.00, '2026-08-19 18:23:40.48105'),
(126, 8, 'credit', 'bonus', 4500.00, 'Performance bonus', 'INIT-012', 8100.00, '2026-08-19 18:23:40.48105'),
(127, 8, 'debit', 'rent', 1100.00, 'Rent payment', 'INIT-013', 7000.00, '2026-08-19 18:23:40.48105'),
(128, 8, 'credit', 'transfer', 9000.00, 'Account top-up', 'INIT-014', 16000.00, '2026-08-19 18:23:40.48105'),
(129, 8, 'credit', 'other', 4000.00, 'Account opening balance', 'INIT-015', 20000.00, '2026-08-19 18:23:40.48105'),
(130, 8, 'debit', 'transfer', 500.00, 'Transfer to Bhavitha', 'TRF-31CDD8E7', 19500.00, '2026-08-19 18:25:42.676609'),
(131, 8, 'debit', 'transfer', 500.00, 'Transfer to Bhavitha', 'TRF-AF69C9DB', 19000.00, '2026-08-20 03:56:37.769602'),
(132, 6, 'debit', 'transfer', 100.00, 'Transfer to Bhavitha', 'TRF-3F1FAD96', 16850.00, '2026-08-20 11:49:26.018361'),
(133, 4, 'debit', 'transfer', 2000.00, 'Transfer to Bhavita', 'TRF-1EDD7E8D', 23000.00, '2026-08-21 05:31:40.476665'),
(134, 3, 'debit', 'transfer', 200.00, 'Transfer to Bhavi', 'TRF-E5A59906', 19800.00, '2026-08-24 08:21:05.080353'),
(135, 8, 'debit', 'transfer', 50.00, 'Transfer to Bhavitha', 'TRF-81A321AE', 18950.00, '2026-08-24 09:05:34.919095'),
(136, 3, 'debit', 'transfer', 600.00, 'Transfer to Bhavi', 'TRF-223F01E1', 19200.00, '2026-08-24 10:12:07.392661'),
(137, 6, 'debit', 'transfer', 200.00, 'Transfer to Bhavitha', 'TRF-F0D90AB0', 16650.00, '2026-08-24 10:22:39.875285'),
(138, 8, 'debit', 'transfer', 200.00, 'Transfer to Bhavitha', 'TRF-FC7D5BFB', 18750.00, '2026-08-24 10:23:34.485781'),
(139, 4, 'debit', 'transfer', 3000.00, 'Transfer to Bhavita', 'TRF-F618256C', 20000.00, '2026-08-24 10:43:14.074824'),
(140, 6, 'debit', 'transfer', 200.00, 'Transfer to Bhavitha', 'TRF-F1BDE77A', 16450.00, '2026-08-25 09:50:33.723303'),
(141, 4, 'debit', 'transfer', 2000.00, 'Transfer to Landlord', 'TRF-07BD622C', 18000.00, '2026-08-31 12:01:52.369591'),
(142, 4, 'debit', 'transfer', 1000.00, 'Transfer to Chinmay', 'TRF-189C28E7', 17000.00, '2026-08-31 14:48:08.785858'),
(143, 4, 'debit', 'transfer', 1.00, 'Transfer to Chinmay', 'TRF-35A190B8', 16999.00, '2026-08-31 14:51:16.440735'),
(144, 4, 'debit', 'transfer', 1000.00, 'Transfer to Chinmay', 'TRF-799BD884', 15999.00, '2026-08-31 14:53:45.587936'),
(145, 4, 'debit', 'transfer', 1000.00, 'Transfer to Chinmay', 'TRF-AA32DFD5', 14999.00, '2026-08-31 15:02:43.604212'),
(146, 3, 'debit', 'transfer', 1000.00, 'Transfer to Bhavi', 'TRF-D0F00C7E', 18200.00, '2026-08-31 15:25:57.769601'),
(147, 3, 'debit', 'transfer', 500.00, 'Transfer to Karu', 'TRF-D5B99E84', 17700.00, '2026-08-31 18:01:09.017102'),
(148, 3, 'debit', 'transfer', 700.00, 'Transfer to Bhavi', 'TRF-4A6F81E1', 17000.00, '2026-08-31 18:33:21.845649'),
(149, 3, 'debit', 'transfer', 5000.00, 'Transfer to Karu', 'TRF-0B298996', 12000.00, '2026-08-31 18:50:41.407088'),
(150, 6, 'debit', 'transfer', 500.00, 'Transfer to Bhavitha', 'TRF-4EC8D06E', 15950.00, '2026-09-01 06:02:57.731397'),
(151, 4, 'debit', 'transfer', 20.00, 'Transfer to Chinmay', 'TRF-DB4D9023', 14979.00, '2026-09-01 06:50:43.049236'),
(152, 9, 'credit', 'salary', 2500.00, 'Salary credit', 'INIT-001', 2500.00, '2026-09-01 06:52:13.274726'),
(153, 9, 'debit', 'rent', 1200.00, 'Rent payment', 'INIT-002', 1300.00, '2026-09-01 06:52:13.274726'),
(154, 9, 'debit', 'groceries', 300.00, 'Groceries', 'INIT-003', 1000.00, '2026-09-01 06:52:13.274726'),
(155, 9, 'credit', 'bonus', 1500.00, 'Bonus payout', 'INIT-004', 2500.00, '2026-09-01 06:52:13.274726'),
(156, 9, 'debit', 'bills', 700.00, 'Utility bill', 'INIT-005', 1800.00, '2026-09-01 06:52:13.274726'),
(157, 9, 'debit', 'transport', 450.00, 'Transport', 'INIT-006', 1350.00, '2026-09-01 06:52:13.274726'),
(158, 9, 'credit', 'salary', 2200.00, 'Salary credit', 'INIT-007', 3550.00, '2026-09-01 06:52:13.274726'),
(159, 9, 'debit', 'shopping', 600.00, 'Shopping', 'INIT-008', 2950.00, '2026-09-01 06:52:13.274726'),
(160, 9, 'debit', 'entertainment', 350.00, 'Streaming services', 'INIT-009', 2600.00, '2026-09-01 06:52:13.274726'),
(161, 9, 'credit', 'transfer', 1200.00, 'Internal transfer', 'INIT-010', 3800.00, '2026-09-01 06:52:13.274726'),
(162, 9, 'debit', 'groceries', 200.00, 'Groceries', 'INIT-011', 3600.00, '2026-09-01 06:52:13.274726'),
(163, 9, 'credit', 'bonus', 4500.00, 'Performance bonus', 'INIT-012', 8100.00, '2026-09-01 06:52:13.274726'),
(164, 9, 'debit', 'rent', 1100.00, 'Rent payment', 'INIT-013', 7000.00, '2026-09-01 06:52:13.274726'),
(165, 9, 'credit', 'transfer', 9000.00, 'Account top-up', 'INIT-014', 16000.00, '2026-09-01 06:52:13.274726'),
(166, 9, 'credit', 'other', 4000.00, 'Account opening balance', 'INIT-015', 20000.00, '2026-09-01 06:52:13.274726'),
(167, 4, 'debit', 'transfer', 1000.00, 'Transfer to Chintmay', 'TRF-CA73EB9B', 13979.00, '2026-09-01 06:53:55.737235'),
(168, 3, 'debit', 'transfer', 50.00, 'Transfer to Karu', 'TRF-2FA30E2E', 11950.00, '2026-09-01 06:57:15.464102'),
(169, 4, 'debit', 'transfer', 100.00, 'Transfer to Bhavitha', 'TRF-3488B28C', 13879.00, '2026-09-01 06:57:59.181053'),
(170, 4, 'debit', 'transfer', 100.00, 'Transfer to Bhavita', 'TRF-D3385106', 13779.00, '2026-09-01 10:00:59.200466'),
(171, 4, 'debit', 'transfer', 200.00, 'Transfer to Bhavita', 'TRF-502947CB', 13579.00, '2026-09-01 10:15:23.100451'),
(172, 3, 'debit', 'transfer', 400.00, 'Transfer to Karu', 'TRF-496E8FE1', 11550.00, '2026-09-01 10:17:49.131521'),
(173, 4, 'debit', 'transfer', 500.00, 'Transfer to Bhavitha', 'TRF-4D8E41A0', 13079.00, '2026-09-01 10:18:12.530493'),
(174, 3, 'debit', 'transfer', 400.00, 'Transfer to Bhavi', 'TRF-D80473AF', 11150.00, '2026-09-01 10:19:09.904348'),
(175, 10, 'credit', 'salary', 25000.00, 'Salary Credit', 'ACC10-SAL-001', 45000.00, '2026-06-04 07:54:54.812041'),
(176, 10, 'debit', 'groceries', 2000.00, 'Grocery Store', 'ACC10-GRC-001', 43000.00, '2026-06-09 07:54:54.812041'),
(177, 10, 'debit', 'rent', 8000.00, 'House Rent', 'ACC10-REN-001', 35000.00, '2026-06-14 07:54:54.812041'),
(178, 10, 'debit', 'bills', 1500.00, 'Electricity Bill', 'ACC10-BIL-001', 33500.00, '2026-06-19 07:54:54.812041'),
(179, 10, 'debit', 'shopping', 3000.00, 'Online Shopping', 'ACC10-SHP-001', 30500.00, '2026-06-24 07:54:54.812041'),
(180, 10, 'debit', 'transport', 600.00, 'Cab Ride', 'ACC10-TRN-001', 29900.00, '2026-06-29 07:54:54.812041'),
(181, 10, 'credit', 'bonus', 5000.00, 'Performance Bonus', 'ACC10-BON-001', 34900.00, '2026-07-04 07:54:54.812041'),
(182, 10, 'debit', 'entertainment', 1200.00, 'Movie Tickets', 'ACC10-ENT-001', 33700.00, '2026-07-09 07:54:54.812041'),
(183, 10, 'debit', 'insurance', 2000.00, 'Health Insurance', 'ACC10-INS-001', 31700.00, '2026-07-14 07:54:54.812041'),
(184, 10, 'debit', 'groceries', 1800.00, 'Supermarket', 'ACC10-GRC-002', 29900.00, '2026-07-24 07:54:54.812041'),
(185, 10, 'credit', 'refund', 1000.00, 'Product Refund', 'ACC10-REF-001', 30900.00, '2026-08-03 07:54:54.812041'),
(186, 10, 'debit', 'bills', 900.00, 'Internet Bill', 'ACC10-BIL-002', 30000.00, '2026-08-08 07:54:54.812041'),
(187, 10, 'debit', 'shopping', 1500.00, 'Clothing Store', 'ACC10-SHP-002', 28500.00, '2026-08-15 07:54:54.812041'),
(188, 10, 'debit', 'transport', 400.00, 'Metro Travel', 'ACC10-TRN-002', 28100.00, '2026-08-23 07:54:54.812041'),
(189, 10, 'debit', 'other', 8100.00, 'Misc Expenses', 'ACC10-OTH-001', 20000.00, '2026-08-31 07:54:54.812041'),
(190, 3, 'debit', 'transfer', 1.00, 'Transfer to Karu', 'TRF-2A66E254', 11149.00, '2026-09-03 06:54:04.004637'),
(191, 3, 'debit', 'transfer', 100.00, 'Transfer to Karo', 'TRF-750F39B0', 11049.00, '2026-09-03 07:24:51.150642'),
(192, 3, 'debit', 'transfer', 100.00, 'Transfer to Karu', 'TRF-69C33CD3', 10949.00, '2026-09-03 07:51:42.299674'),
(193, 11, 'credit', 'salary', 200000.00, 'Executive Salary', 'EXE-SAL-001', 220000.00, '2026-06-05 08:00:15.414347'),
(194, 11, 'debit', 'business_travel', 25000.00, 'International Flight', 'EXE-TRV-001', 195000.00, '2026-06-10 08:00:15.414347'),
(195, 11, 'debit', 'corporate_dining', 12000.00, 'Board Dinner', 'EXE-DIN-001', 183000.00, '2026-06-15 08:00:15.414347'),
(196, 11, 'credit', 'bonus', 50000.00, 'Annual Bonus', 'EXE-BON-001', 233000.00, '2026-06-20 08:00:15.414347'),
(197, 11, 'debit', 'conference', 18000.00, 'Leadership Summit', 'EXE-CONF-001', 215000.00, '2026-06-25 08:00:15.414347'),
(198, 11, 'debit', 'transport', 8000.00, 'Chauffeur Service', 'EXE-TRN-001', 207000.00, '2026-06-30 08:00:15.414347'),
(199, 11, 'credit', 'reimbursement', 15000.00, 'Travel Reimbursement', 'EXE-REB-001', 222000.00, '2026-07-05 08:00:15.414347'),
(200, 11, 'debit', 'investment', 30000.00, 'Equity Investment', 'EXE-INV-001', 192000.00, '2026-07-10 08:00:15.414347'),
(201, 11, 'debit', 'shopping', 10000.00, 'Luxury Watch', 'EXE-SHP-001', 182000.00, '2026-07-15 08:00:15.414347'),
(202, 11, 'debit', 'bills', 6000.00, 'Office Utilities', 'EXE-BIL-001', 176000.00, '2026-07-25 08:00:15.414347'),
(203, 11, 'credit', 'transfer', 20000.00, 'Funds From HQ', 'EXE-TRF-001', 196000.00, '2026-08-04 08:00:15.414347'),
(204, 11, 'debit', 'entertainment', 7000.00, 'Corporate Event', 'EXE-ENT-001', 189000.00, '2026-08-09 08:00:15.414347'),
(205, 11, 'debit', 'insurance', 9000.00, 'Premium Insurance', 'EXE-INS-001', 180000.00, '2026-08-16 08:00:15.414347'),
(206, 11, 'credit', 'salary', 200000.00, 'Executive Salary', 'EXE-SAL-002', 380000.00, '2026-08-24 08:00:15.414347'),
(207, 11, 'debit', 'other', 160000.00, 'Corporate Expenses', 'EXE-OTH-001', 220000.00, '2026-09-01 08:00:15.414347'),
(208, 7, 'debit', 'transfer', 5000.00, 'Transfer to Jina', 'TRF-2B5F2326', 45000.00, '2026-09-03 08:17:27.373062'),
(209, 12, 'credit', 'salary', 2500.00, 'Salary credit', 'INIT-001', 2500.00, '2026-09-03 08:22:05.391509'),
(210, 12, 'debit', 'rent', 1200.00, 'Rent payment', 'INIT-002', 1300.00, '2026-09-03 08:22:05.391509'),
(211, 12, 'debit', 'groceries', 300.00, 'Groceries', 'INIT-003', 1000.00, '2026-09-03 08:22:05.391509'),
(212, 12, 'credit', 'bonus', 1500.00, 'Bonus payout', 'INIT-004', 2500.00, '2026-09-03 08:22:05.391509'),
(213, 12, 'debit', 'bills', 700.00, 'Utility bill', 'INIT-005', 1800.00, '2026-09-03 08:22:05.391509'),
(214, 12, 'debit', 'transport', 450.00, 'Transport', 'INIT-006', 1350.00, '2026-09-03 08:22:05.391509'),
(215, 12, 'credit', 'salary', 2200.00, 'Salary credit', 'INIT-007', 3550.00, '2026-09-03 08:22:05.391509'),
(216, 12, 'debit', 'shopping', 600.00, 'Shopping', 'INIT-008', 2950.00, '2026-09-03 08:22:05.391509'),
(217, 12, 'debit', 'entertainment', 350.00, 'Streaming services', 'INIT-009', 2600.00, '2026-09-03 08:22:05.391509'),
(218, 12, 'credit', 'transfer', 1200.00, 'Internal transfer', 'INIT-010', 3800.00, '2026-09-03 08:22:05.391509'),
(219, 12, 'debit', 'groceries', 200.00, 'Groceries', 'INIT-011', 3600.00, '2026-09-03 08:22:05.391509'),
(220, 12, 'credit', 'bonus', 4500.00, 'Performance bonus', 'INIT-012', 8100.00, '2026-09-03 08:22:05.391509'),
(221, 12, 'debit', 'rent', 1100.00, 'Rent payment', 'INIT-013', 7000.00, '2026-09-03 08:22:05.391509'),
(222, 12, 'credit', 'transfer', 9000.00, 'Account top-up', 'INIT-014', 16000.00, '2026-09-03 08:22:05.391509'),
(223, 12, 'credit', 'other', 4000.00, 'Account opening balance', 'INIT-015', 20000.00, '2026-09-03 08:22:05.391509'),
(224, 10, 'debit', 'transfer', 1000.00, 'Transfer to Kavitha', 'TRF-BD334850', 29000.00, '2026-09-03 11:54:01.621649');

SELECT setval('transactions_id_seq', (SELECT MAX(id) FROM transactions));

-- Sample cheque deposit requests, in different statuses
INSERT INTO cheque_requests (request_id, phone_number, bank_name, branch, payee, amount_in_figures, amount_in_words, cheque_number, signatory, status, created_at) VALUES
('CHQ-A1B2C3D4', '919843222825', 'Finacle Banking', 'London Central', 'Ramprasath G', '500.00', 'Five Hundred Pounds Only', '000123', 'A. Patel', 'COMPLETED', NOW() - INTERVAL '10 days'),
('CHQ-E5F6G7H8', '919843222825', 'Finacle Banking', 'London Central', 'Ramprasath G', '1200.00', 'One Thousand Two Hundred Pounds Only', '000124', 'A. Patel', 'PENDING', NOW() - INTERVAL '1 day'),
('CHQ-J9K1L2M3', '918956140132', 'Barclays', 'Manchester', 'Priya P', '75.00', 'Seventy Five Pounds Only', '000045', 'R. Khan', 'REJECTED', NOW() - INTERVAL '5 days');

-- Saved beneficiaries — synced from live database snapshot, 2026-09-04
INSERT INTO beneficiaries (phone_number, beneficiary_name, account_number, bank_name, created_at) VALUES
('447818658034', 'Priya Sharma',                        'GB29FNCL60161331926819', 'Finacle Banking', '2026-08-18 08:51:47.934207'),
('447818658034', 'Alex Morgan',                          'GB77FNCL29001847502211', 'Finacle Banking', '2026-08-18 08:51:47.934207'),
('919080745760', 'Rahul Verma',                          'GB14FNCL74208891736642', 'Finacle Banking', '2026-08-18 08:51:47.934207'),
('919080745760', 'Emma Wilson',                          'GB05FNCL13590027461938', 'Finacle Banking', '2026-08-18 08:51:47.934207'),
('917012821406', 'My Landlord If Balance Is More Than',  'ACCOUNTNUMBERIS4639356', NULL,              '2026-08-18 09:21:42.239804'),
('917829918363', 'Bhavitha',                              '000234678910',           NULL,              '2026-08-19 06:39:14.376073'),
('917331122909', 'Bhavitha',                              'INFISCO1234567',         NULL,              '2026-08-19 06:41:43.178099'),
('917829918363', 'Bhavita',                               'INF937388221',           NULL,              '2026-08-19 08:57:49.072382'),
('917829918363', 'Chinmay',                               '3688992W22334',          NULL,              '2026-08-19 08:59:49.4777'),
('919843222825', 'Jeena',                                 'FNCL000000000004',       'Finacle Banking', '2026-08-19 15:16:33.420979'),
('919843222825', 'Bhavitha',                              'FNCLBENEFACC00005',      'Finacle Banking', '2026-08-19 15:16:33.420979'),
('919843222825', 'Chinmai',                               'FNCLBENEFACC00006',      'Finacle Banking', '2026-08-19 15:16:33.420979'),
('919384148239', 'Bhavi',                                 '123456789',              NULL,              '2026-08-24 08:20:46.524316'),
('917829918363', 'Landlord',                              'ACC32547974',            NULL,              '2026-08-31 12:00:55.968051'),
('919384148239', 'Karu',                                  '1212167890',             NULL,              '2026-08-31 18:00:43.764296'),
('917829918363', 'Chintmay',                              'WQ1294874',              NULL,              '2026-09-01 06:53:35.55601'),
('919384148239', 'Bobby',                                 '1234567891',             NULL,              '2026-09-01 10:22:10.596508'),
('919384148239', 'Karo',                                  '2222222222',             NULL,              '2026-09-03 07:24:41.301344'),
('919843222825', 'Jina',                                  '1339829',                NULL,              '2026-09-03 08:17:07.438541'),
('916384354499', 'Kavitha',                               '123446788',              NULL,              '2026-09-03 11:52:05.528711');