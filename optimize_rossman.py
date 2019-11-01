"""Optimize fastai tabular learner for Rossman without Pruning Callback."""
import argparse
import datetime
from functools import partial
import logging
import os

from fastai.tabular import *
from fastai.callbacks import TrackerCallback


# create logger with 'spam_application'
logger = logging.getLogger('optuna-fastai')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('optuna.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

assert torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--pruning', '-p', action='store_true', default=False)
args = parser.parse_args()
logger.info('Pruning is enabled!' if args.pruning else 'Pruning is disabled!')


# Callback for Pruning.
class FastAIPruningCallback(TrackerCallback):
    def __init__(self, learn, trial, monitor):
        # type: (Learner, optuna.trial.Trial, str) -> None

        super(FastAIPruningCallback, self).__init__(learn, monitor)

        self.trial = trial

    def on_epoch_end(self, epoch, **kwargs):
        # type: (int, Any) -> None

        value = self.get_monitor_value()
        if value is None:
            return

        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            message = 'Trial was pruned at epoch {}.'.format(epoch)
            raise optuna.structs.TrialPruned(message)


logger.debug("Loading datasets...")
path = Path("./data")
train_df = pd.read_pickle(path/'train_clean')
test_df = pd.read_pickle(path/'test_clean')
n = len(train_df)
procs= [FillMissing, Categorify, Normalize]
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h',
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

dep_var = 'Sales'
df = train_df[cat_vars + cont_vars + [dep_var, 'Date']].copy()
cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()
valid_idx = range(cut)

logger.debug('Constructing DataBunch')
data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))
                .databunch())

max_log_y = np.log(np.max(train_df['Sales']) * 1.2)
y_range = torch.tensor([0, max_log_y]).to(torch.device('cuda:0'))


def objective(trial):
    # Objective is `exp_rmspe`.
    n_layers = trial.suggest_int('n_layers', 1, 5)
    layers = []
    ps = []
    for i in range(n_layers - 1):
        # Number of units
        n_units = trial.suggest_categorical(
            'n_units_layer_{}'.format(i), [800, 900, 1000, 1100, 1200])
        layers.append(n_units)
        # Dropout ratio
        p = trial.suggest_discrete_uniform(
            'dropout_p_layer_{}'.format(i), 0, 1, 0.05)
        ps.append(p)

    emb_drop = trial.suggest_discrete_uniform('emb_drop', 0, 1, 0.05)
    callback_fns = []
    if args.pruning:
        logger.info("FastAIPruningCallback is registered to callback_fns")
        callback_fns.append(
            partial(FastAIPruningCallback, trial=trial, monitor='exp_rmspe')
        )
    learn = tabular_learner(
        data, layers=layers, ps=ps, emb_drop=emb_drop, y_range=y_range,
        metrics=exp_rmspe, callback_fns=callback_fns, silent=True)

    learn.fit_one_cycle(5, 1e-3, wd=0.2)

    return learn.validate()[-1].item()


if __name__ == '__main__':
    import json
    import optuna

    outpath = "./out/{}".format("pruning" if args.pruning else "vanilla")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    logger.info("Save outputs to {}".format(outpath))

    logger.info("Create study...")
    study = optuna.create_study(
        study_name='fastai-optuna-no-pruning',
        direction='minimize')

    start = datetime.datetime.now()
    study.optimize(objective, n_trials=100)
    end = datetime.datetime.now()

    if args.pruning:
        pruned_trials = [
            t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED
        ]
        complete_trials = [
            t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE
        ]
        print('Study statistics: ')
        print('  Number of finished trials: ', len(study.trials))
        print('  Number of pruned trials: ', len(pruned_trials))
        print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))

    best_params = study.best_params
    best_params['exp_rmspe'] = study.best_value

    with open('./best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(outpath, 'trials.csv'))
    logger.info('Dumping results to data frame')

    print(f'total time is {(end - start).total_seconds()} seconds')
