# Benchmarks

These reports compare deep-river models against strong streaming baselines on standard datasets. Use them to understand trade-offs between predictive quality, model complexity, and update behavior in online scenarios.

Each benchmark chapter includes dataset context, model configurations, and visual summaries to support reproducible comparisons.

## How to read these reports

- Treat the tables as task-specific comparisons, not universal model rankings.
- Compare predictive metrics together with memory and update time; deep models often trade speed for flexibility.
- Pay attention to stream order and dataset context, because online results depend on when each item arrives.

## Reports

- [Binary Classification](binary-classification/index.md)
- [Multiclass Classification](multiclass-classification/index.md)
- [Regression](regression/index.md)
- [Forecasting](forecasting/index.md)
