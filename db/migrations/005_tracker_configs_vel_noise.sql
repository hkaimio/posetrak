BEGIN;
ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_std REAL;
PRAGMA user_version = 6;
COMMIT;
