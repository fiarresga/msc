import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import bellatrex as btrex

seed = 7

all_times_df = pd.read_csv('predictors_plus_target_calibrated.csv')
all_times_df = all_times_df.drop(['p6177_i0', 'p90004', 'p4079_i0_a0', 'p94_i0_a0', 'p4080_i0_a0', 'p93_i0_a0', 'p20002_i0_a0', 'p6177_i0', 'p30760_i0', 'p30870_i0'], axis=1)
all_times_df = all_times_df.dropna()
print('Number of Individuals ', len(all_times_df))

label_encoder = LabelEncoder()
all_times_df['activity_class'] = label_encoder.fit_transform(all_times_df['activity_class'])

all_times_df['p31'] = pd.factorize(all_times_df['p31'])[0]
all_times_df['p2306_i0'] = pd.factorize(all_times_df['p2306_i0'])[0]
all_times_df['p2443_i0'] = pd.factorize(all_times_df['p2443_i0'])[0]

X = all_times_df[['Mean', 'STD', 'skewR', 'kurtR', 'perc95', 'perc5', 'dPerc', 'Peak power', 'PPFd', 'Entropy', 'PF_sum', 'P_sum', 'p31','Steps']]
y = all_times_df['activity_class']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

print("Dataset created")
##########


rf_classifier = RandomForestClassifier(random_state=seed)
rf_classifier.fit(X_train_full, y_train_full)

print("FITTED!")
print(btrex.__version__)

from bellatrex.wrapper_class import pack_trained_ensemble

# Pretrained RF model should be packed as a list of dicts with the function below.
clf_packed = pack_trained_ensemble(rf_classifier)

from bellatrex import BellatrexExplain
from bellatrex.utilities import predict_helper

#fit RF here. The hyperparameters for fitting the explanation are given
# compatible with trained ensemble model clf, and with packed dictionary as in clf_packed
Btrex_fitted = BellatrexExplain(clf_packed, set_up='auto',
                                p_grid={"n_clusters": [1, 2, 3]},
                                verbose=3).fit(X_train_full, y_train_full)

N_TEST_SAMPLES = 2
for i in range(N_TEST_SAMPLES):

    print(f"Explaining sample i={i}")

    y_train_pred = predict_helper(rf_classifier, X_train_full) # calls, predict or predict_proba, depending on the underlying model

    tuned_method = Btrex_fitted.explain(X_test, i)

    tuned_method.plot_overview(plot_gui=False,
                               show=True)


    tuned_method.plot_visuals(plot_max_depth=5,
                              preds_distr=y_train_pred,
                              conf_level=0.9,
                              tot_digits=4)
    plt.show()

