DATASET="tourism"
for METHOD in HierE2E DeepVAR DeepVARPlus ETS_NaiveBU ARIMA_NaiveBU ETS_MINT_shr ETS_MINT_ols ARIMA_MINT_shr ARIMA_MINT_ols ETS_ERM ARIMA_ERM PERMBU_MINT; 
do
    echo "================================"$METHOD"================================"
    python experiments/run_experiment_with_best_hps.py --dataset $DATASET --method $METHOD
done
