# Input table

The scripts read `data/research_layers/health_twin_base_v1.csv.gz`, one row per turbine and 10-min interval, and check its SHA-256 hash against `output_sha256` in `data/research_layers/health_twin_base_v1_manifest.json`.

Signal descriptions are provisional interpretations of the operator's signal labels; no manufacturer signal dictionary was available. Values are in the units of the SCADA export.

| Column | Content |
|---|---|
| `timestamp` | Start of the 10-min interval |
| `turbine_id` | Turbine label, `T01` to `T16` |
| `has_temperature_record` | True if the temperature table has a row for this turbine at exactly this timestamp |
| `active_power_mean` | Active power, kW |
| `wind_speed_mean` | Primary wind speed, m/s |
| `generator_speed_mean` | Generator speed, rpm |
| `main_shaft_speed_mean` | Main-shaft speed |
| `pitch_a_mean`, `pitch_b_mean`, `pitch_c_mean` | Blade pitch positions |
| `power_reference_end` | Power reference at the end of the interval, kW |
| `local_ambient_temp_mean` | Turbine-local ambient temperature (proxy) |
| `met_temperature_mean` | Temperature of the independent meteorological station |
| `met_wind_speed_mean`, `met_wind_direction_mean`, `met_humidity_mean`, `met_pressure_mean` | Other meteorological-station signals, used only by the meteorological quality-gate audit |
| `gear_oil_temp_mean` | Gear-oil temperature (primary target) |
| `gen_bearing_g_temp_mean` | Generator-bearing-G temperature (primary target) |
| `gen_bearing_r_temp_mean` | Generator-bearing-R temperature (primary target) |
| `main_bearing_temp_mean` | Main-bearing temperature (exploratory target) |
| `hydraulic_oil_temp_mean` | Hydraulic-oil temperature (negative-control target) |

Meteorological-station values are the same for all turbines at a given timestamp.
