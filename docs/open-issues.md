## State

- why root_velocity_ is VectorXd, not VEctor3D (2026-01-26)

- Skeleton::active_dof() does not count root position & orientation. Should it? Or is it only part of the tracker state?

- ObservationsSet still has the method get_all_at_time() which seems to cause issues. We really should use get_all_in_range always.
