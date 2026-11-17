# Input table

The scripts read `data/research_layers/health_twin_base_v1.csv.gz`, one row per turbine and 10-min interval, and check its SHA-256 hash against `output_sha256` in `data/research_layers/health_twin_base_v1_manifest.json`.

The operator confirmed the descriptions and Celsius units of the five target temperatures and that the turbine-local ambient signal measures outdoor temperature outside the nacelle. A complete manufacturer signal dictionary was not available; other values are in the units of the SCADA export. The 2021–2022 table has the same columns.

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
| `local_ambient_temp_mean` | Turbine-local outdoor temperature, °C (the local-ambient proxy of the L folds and of the 2021–2022 confirmation) |
| `met_temperature_mean` | Temperature of the independent meteorological station |
| `met_wind_speed_mean`, `met_wind_direction_mean`, `met_humidity_mean`, `met_pressure_mean` | Other meteorological-station signals, used only by the meteorological quality-gate audit |
| `gear_oil_temp_mean` | Gear-oil temperature, °C (primary target) |
| `gen_bearing_g_temp_mean` | Generator-bearing temperature, gearbox side (drive end), °C (primary target) |
| `gen_bearing_r_temp_mean` | Generator-bearing temperature, rear side (non-drive end), °C (primary target) |
| `main_bearing_temp_mean` | Rotor main-bearing temperature, °C (exploratory target) |
| `hydraulic_oil_temp_mean` | Hydraulic-oil temperature, °C (comparison channel) |

Meteorological-station values are the same for all turbines at a given timestamp.
