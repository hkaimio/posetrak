-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration: session schema v4 → v5
-- Adds nis_value and nis_dof columns to tracking_results for UKF consistency
-- monitoring. nis_value is the Normalized Innovation Squared (should follow a
-- chi-squared distribution with nis_dof degrees of freedom when the filter is
-- consistent). Existing rows get NULL (no historical data available).

BEGIN;

ALTER TABLE tracking_results ADD COLUMN nis_value REAL;
ALTER TABLE tracking_results ADD COLUMN nis_dof   INTEGER;

PRAGMA user_version = 5;

COMMIT;
