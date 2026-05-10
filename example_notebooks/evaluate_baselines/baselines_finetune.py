import optuna
import pickle
from functools import partial
from pathlib import Path

from simulator.simulation.modules import Campaign
from simulator.simulation.utils_visualization import data_prep_vis, plot_history_article
from simulator.simulation.simulate import simulate_campaign
from simulator.validation.check_results import autobidder_check

# import baselines:
from simulator.model.linear_bidder import LinearBidder
from simulator.model.ta_pid import TAPIDBidder
from simulator.model.m_pid import MPIDBidder
from simulator.model.mystique import Mystique
from simulator.model.broi_bidder import BROI
from simulator.model.traffic import Traffic

# TODO нужно уточнить все таки должно быть SCR написано и оно же максимизируется?

class BaseLineTrainer:
    def get_metric_score(self, metric: str):
        if metric == "CPC_REL":
            return 0
        elif metric == "RMSE":
            return 1
        elif metric == "SCR":
            return 2

    def __init__(
        self,
        data_config,
        metric: str,
        auction_mode: str,
        base_params_subfolder: str,
        random_state: int = 42,
        params_dir: Path | str | None = None,
        n_jobs: int | None = None,
    ):
        self.data_config = data_config
        self.metric = metric
        self.auction_mode = auction_mode
        self.random_state = random_state
        self.params_dir = None if params_dir is None else Path(params_dir)
        self.n_jobs = n_jobs
        if self.metric not in ["CPC_REL", "RMSE", "SCR"]:
            raise Exception("Wrong metric objective")
        self.score_indx = self.get_metric_score(metric)
        self.best_params_subfolder = base_params_subfolder
    
    

    def _get_params_path(self, model_name: str) -> str:
        if self.params_dir is not None:
            return str(self.params_dir / f"{model_name}_{self.metric.lower()}_{self.auction_mode}.pkl")
        return f'best_params/{self.best_params_subfolder}/{model_name}_{self.metric.lower()}_{self.auction_mode}.pkl'

    def get_params_path(self, model_name: str) -> str:
        return self._get_params_path(model_name)

    def _study_direction(self) -> str:
        return 'maximize' if self.metric == 'SCR' else 'minimize'

    def _study_n_jobs(self, default: int) -> int:
        return default if self.n_jobs is None else int(self.n_jobs)


    def objective_linear(self, trial, eval: bool = False):
        if eval:
            with open(self._get_params_path('linear'), 'rb') as f:
                params_dict = pickle.load(f)
            campaigns_path = self.data_config['test']['campaigns_path']
            stats_path = self.data_config['test']['stats_path']
            coef = params_dict['coef']
            lower_clip = params_dict['lower_clip']
            upper_clip = params_dict['upper_clip']
            factor = params_dict['factor']
        else:
            campaigns_path = self.data_config['train']['campaigns_path']
            stats_path = self.data_config['train']['stats_path']
            coef = trial.suggest_float('coef', 0.001, 0.999, log=True)
            lower_clip = trial.suggest_int('lower_clip', 1, 20, log=True)
            upper_clip = trial.suggest_int('upper_clip', 1, 20, log=True)
            factor = trial.suggest_float('factor', 1.1, 10.0, log=True)

        res = autobidder_check(
            bidder=LinearBidder,
            params={
                "input_campaigns": campaigns_path,
                "input_stats": stats_path,
                "cold_start_coef": coef,
                "lower_clip": lower_clip,
                "upper_clip": upper_clip,
                "factor": factor
            },
            auction_mode=self.auction_mode,
        )
        print(f"CPC_REL: {res['score'][0]}, rmse: {res['score'][1]}, SCR: {res['score'][2]}")
        if eval:
            return res['score']
        return res['score'][self.score_indx]

    def opt_search_linear(self, n_trials=100):
        dict_path = self._get_params_path('linear')
        # Create pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )

        # Create study with pruner
        study = optuna.create_study(
            direction=self._study_direction(),
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Optimization
        study.optimize(
            partial(self.objective_linear),
            n_trials=n_trials,
            catch=(Exception,)
        )

        print('Best trial:')
        trial = study.best_trial

        print(f'Value: {trial.value}')
        print('Params: ')
        params_dict = dict()
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
            params_dict[key] = value
        # save the best params to the file
        Path(dict_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dict_path, 'wb') as f:
            pickle.dump(params_dict, f)
        return study
    

    def objective_tapid(self, trial, eval: bool = False):
        if eval:
            with open(self._get_params_path('ta_pid'), 'rb') as f:
                params_dict = pickle.load(f)
            campaigns_path = self.data_config['test']['campaigns_path']
            stats_path = self.data_config['test']['stats_path']
            k_dict = {'k_p': params_dict['k_p1'], 'k_i': params_dict['k_i1'], 'k_d': params_dict['k_d1']}
            coef = params_dict['coef']
        else:
            campaigns_path = self.data_config['train']['campaigns_path']
            stats_path = self.data_config['train']['stats_path']
            k_p1 = trial.suggest_float('k_p1', 1e-4, 1, log=True)
            k_i1 = trial.suggest_float('k_i1', 1e-4, 1, log=True)
            k_d1 = trial.suggest_float('k_d1', 1e-4, 1, log=True)
            coef = trial.suggest_float('coef', 0.1, 0.9, log=True)
            k_dict = {'k_p': k_p1, 'k_i': k_i1, 'k_d': k_d1}

        res = autobidder_check(
            bidder=TAPIDBidder,
            params={
                "input_campaigns": campaigns_path,
                "input_stats": stats_path,
                "k_dict": k_dict,
                # TAPIDBidder uses key "coef" (not cold_start_coef like Linear/MPID).
                "coef": coef,
            },
            auction_mode=self.auction_mode,
        )
        print(f"CPC_REL: {res['score'][0]}, rmse: {res['score'][1]}, SCR: {res['score'][2]}")
        if eval:
            return res['score']
        return res['score'][self.score_indx]

    def opt_search_tapid(self, n_trials):
        dict_path = self._get_params_path('ta_pid')
        study = optuna.create_study(
            direction=self._study_direction(),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(partial(self.objective_tapid), n_trials=n_trials)

        print('Best trial:')
        trial = study.best_trial

        print(f'Value: {trial.value}')
        print('Params: ')
        params_dict = dict()
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
            params_dict[key] = value
        # save the best params to the file
        Path(dict_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dict_path, 'wb') as f:
            pickle.dump(params_dict, f)
        return study

    
    def objective_mpid(self, trial, eval: bool = False):
        if eval:
            with open(self._get_params_path('m_pid'), 'rb') as f:
                params_dict = pickle.load(f)
            campaigns_path = self.data_config['test']['campaigns_path']
            stats_path = self.data_config['test']['stats_path']
            k_dict = {
                'k_p': (params_dict['k_p1'], params_dict['k_p2']),
                'k_i': (params_dict['k_i1'], params_dict['k_i2']),
                'k_d': (params_dict['k_d1'], params_dict['k_d2']),
            }
            alpha, beta = params_dict['alpha'], params_dict['beta']
            coef = params_dict['coef']
            lower_clip = params_dict['lower_clip']
            upper_clip = params_dict['upper_clip']
            bid_factor = params_dict['bid_factor']
        else:
            campaigns_path = self.data_config['train']['campaigns_path']
            stats_path = self.data_config['train']['stats_path']
            k_p1 = trial.suggest_float('k_p1', 1e-4, 1, log=True)
            k_p2 = trial.suggest_float('k_p2', 1e-4, 1, log=True)
            k_i1 = trial.suggest_float('k_i1', 1e-4, 1, log=True)
            k_i2 = trial.suggest_float('k_i2', 1e-4, 1, log=True)
            k_d1 = trial.suggest_float('k_d1', 1e-4, 1, log=True)
            k_d2 = trial.suggest_float('k_d2', 1e-4, 1, log=True)
            alpha = trial.suggest_float('alpha', 0.1, 0.9, log=True)
            beta = trial.suggest_float('beta', 0.1, 0.9, log=True)
            coef = trial.suggest_float('coef', 0.1, 0.9, log=True)
            lower_clip = trial.suggest_float('lower_clip', 0.1, 0.9, log=True)
            upper_clip = trial.suggest_float('upper_clip', 1.5, 10, log=True)
            bid_factor = trial.suggest_float('bid_factor', 0.5, 20.0, log=True)
            k_dict = {'k_p': (k_p1, k_p2), 'k_i': (k_i1, k_i2), 'k_d': (k_d1, k_d2)}

        res = autobidder_check(
            bidder=MPIDBidder,
            params={
                "input_campaigns": campaigns_path,
                "input_stats": stats_path,
                "k_dict": k_dict,
                "correction": [alpha, beta],
                'cold_start_coef': coef,
                'lower_clip': lower_clip,
                'upper_clip': upper_clip,
                'bid_factor': bid_factor,
            },
            auction_mode=self.auction_mode,
        )
        print(f"CPC_REL: {res['score'][0]}, rmse: {res['score'][1]}, SCR: {res['score'][2]}")
        if eval:
            return res['score']
        return res['score'][self.score_indx]

    def opt_search_mpid(self, n_trials):
        dict_path = self._get_params_path('m_pid')
        study = optuna.create_study(
            direction=self._study_direction(),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(partial(self.objective_mpid), n_trials=n_trials, n_jobs=self._study_n_jobs(6))

        print('Best trial:')
        trial = study.best_trial

        print(f'  Value: {trial.value}')
        print('  Params: ')
        params_dict = dict()
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
            params_dict[key] = value
        # save the best params to the file
        Path(dict_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dict_path, 'wb') as f:
            pickle.dump(params_dict, f)
        return study


    def objective_mystique(self, trial, eval: bool = False):
        if eval:
            with open(self._get_params_path('mystique'), 'rb') as f:
                params_dict = pickle.load(f)
            campaigns_path = self.data_config['test']['campaigns_path']
            stats_path = self.data_config['test']['stats_path']
            pf0 = params_dict['pf0']
            C_max = params_dict['C_max']
            C_min = params_dict['C_min']
            E_max = params_dict['E_max']
            E_gmc = params_dict['E_gmc']
        else:
            campaigns_path = self.data_config['train']['campaigns_path']
            stats_path = self.data_config['train']['stats_path']
            pf0 = trial.suggest_float('pf0', 1, 200, log=True)
            C_max = trial.suggest_float('C_max', 1, 100, log=True)
            C_min = trial.suggest_float('C_min', 1e-4, 1, log=True)
            E_max = trial.suggest_float('E_max', 1, 100, log=True)
            E_gmc = trial.suggest_float('E_gmc', 1, 100, log=True)

        res = autobidder_check(
            bidder=Mystique,
            params={
                "input_campaigns": campaigns_path,
                "input_stats": stats_path,
                "pf0": pf0,
                "C_max": C_max,
                'C_min': C_min,
                'E_max': E_max,
                'E_gmc': E_gmc,
            },
            auction_mode=self.auction_mode,
        )
        print(f"CPC_REL: {res['score'][0]}, rmse: {res['score'][1]}, SCR: {res['score'][2]}")
        if eval:
            return res['score']
        return res['score'][self.score_indx]

    def opt_search_mystique(self, n_trials):
        dict_path = self._get_params_path('mystique')
        study = optuna.create_study(
            direction=self._study_direction(),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(partial(self.objective_mystique), n_trials=n_trials, n_jobs=self._study_n_jobs(6))

        print('Best trial:')
        trial = study.best_trial

        print(f'  Value: {trial.value}')
        print('  Params: ')
        params_dict = dict()
        for key, value in trial.params.items():
            print(f'{key}: {value}')
            params_dict[key] = value
        # save the best params to file
        Path(dict_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dict_path, 'wb') as f:
            pickle.dump(params_dict, f)
        return study
    

    def objective_broi(self, trial, eval: bool = False):
        if eval:
            with open(self._get_params_path('broi'), 'rb') as f:
                params_dict = pickle.load(f)
            campaigns_path = self.data_config['test']['campaigns_path']
            stats_path = self.data_config['test']['stats_path']
            ro = params_dict['ro']
            v_bar = params_dict['v_bar']
        else:
            campaigns_path = self.data_config['train']['campaigns_path']
            stats_path = self.data_config['train']['stats_path']
            ro = trial.suggest_float('ro', 0.01, 200, log=True)
            v_bar = trial.suggest_float('v_bar', 0.01, 200, log=True)

        res = autobidder_check(
            bidder=BROI,
            params={
                "input_campaigns": campaigns_path,
                "input_stats": stats_path,
                'ro': ro,
                'v_bar': v_bar
            },
            auction_mode=self.auction_mode,
        )
        print(f"CPC_REL: {res['score'][0]}, rmse: {res['score'][1]}, SCR: {res['score'][2]}")
        if eval:
            return res['score']
        return res['score'][self.score_indx]

    def opt_search_broi(self, n_trials=100):
        dict_path = self._get_params_path('broi')
        # Create pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        # Create study with pruner (in-memory, like other opt_search_*; we dropped
        # persistent sqlite: with catch=(Exception,) a full run of failures left
        # no best row and RDB raised ValueError "Record does not exist").
        study = optuna.create_study(
            direction=self._study_direction(),
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        # Optimization
        study.optimize(
            partial(self.objective_broi),
            n_trials=n_trials,
            catch=(Exception,)
        )
        # At least one trial must have completed; otherwise best_trial is undefined.
        success = [
            t
            for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
        ]
        if not success:
            raise RuntimeError(
                "opt_search_broi: no successful Optuna trials (all may have been caught "
                "as exceptions in objective_broi). Check logs; verify train campaigns/stats paths."
            )
        print('Best trial:')
        trial = study.best_trial
        print(f'  Value: {trial.value}')
        print('  Params: ')
        params_dict = dict()
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
            params_dict[key] = value
        # save the best params to the file
        Path(dict_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dict_path, 'wb') as f:
            pickle.dump(params_dict, f)
        return study
