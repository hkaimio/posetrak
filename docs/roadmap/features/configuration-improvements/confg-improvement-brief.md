# Tracker configuration improvements

## Problems

1. The configuration dialog is way too complex. All parameters are listed in a single monolithic pane as a list
2. Config changes must be entered every time a tracking is started. The schema & CLI support (actually require) saving a configuration and using it in multiple tracking runs by its id but this is not visible in UI


Proposed approach

* Add support for saving & loading configuration. Ideally, there should be
  - a default configuration per session, capture & trial (if e.g. trial does not have default config, use capture's config etc). On trial page there should be button/menu item to edit the default config
  - option to save & load a named configuration
  - when new tracking is started, possible to edit the default or loaded config which creates tracking run specific configuration (which can later be saved with name)

* Organize the configuration dialog
  - Add multiple pages as vertical tabs
  - First tab show summary of the full configuration (like the tracking run side bar now but bit more verbose)
  - Other tabs have logically grouped settings nicely laid out, with tooltips that explain each setting
  - No spin buttons without good reasoning - numeric fields are usually the best option

  ## Other realted, not deirsctoy config specific but relevalt for UI desing

  * Currently persons are detection  specific, I'd like eventually move their definition tor trial and/or capture level (as they usually are same). So that user could define the persons in trial, define what skeleton to use for each and then when starting trackign run select which persons to track (and if needed override the skeleton too)

  * CLI shouls support similar model
