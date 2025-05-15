import os
import numpy as np
from typing import Optional, List, Dict
from omegaconf import DictConfig, open_dict


class Logger:
    def __init__(self, cfg: DictConfig, verbose: bool = True) -> None:
        self.cfg = cfg
        self.path = cfg.path
        self.verbose = verbose
        self.unique_id = cfg.unique_id

        self.log_filepath = os.path.join(self.path, 'test_result.csv')
        self.log_iter_path = None
        self.log_fold_path = None

        self.all_results = []
        self.all_fold_results = []

    def init_logging(self) -> None:
        with open(self.log_filepath, 'a') as f:
            f.write('Iter,AUC,ACC,SEN,SPC\n')

    def init_iter_logging(self, iteration: int) -> None:
        iter_dir = os.path.join(self.path, f'Iter_{iteration + 1}')
        os.makedirs(iter_dir, exist_ok=True)
        self.log_iter_path = os.path.join(iter_dir, f'iter{iteration + 1}_test_result.csv')
        with open(self.log_iter_path, 'a') as f:
            f.write('Fold,AUC,ACC,SEN,SPC\n')

    def init_fold_logging(self, iteration: int, fold: int) -> None:
        self.log_fold_path = os.path.join(
            self.path, f'Iter_{iteration + 1}', f'iter{iteration + 1}_fold{fold + 1}_log.csv'
        )
        with open(self.log_fold_path, 'a') as f:
            f.write('Epoch,Train Loss,Train ACC,Valid Loss,Valid AUC,Valid ACC,Test Loss,'
                    'Test AUC,Test ACC,Test SEN,Test SPC\n')

    def log_epoch(
            self, epoch: int, train: Dict[str, float],
            valid: Dict[str, float], test: Dict[str, float]
    ) -> None:
        with open(self.log_fold_path, 'a') as f:
            f.write(
                f"{epoch + 1},{train['loss']:.5f},{train['acc']:.5f},"
                f"{valid['loss']:.5f},{valid['auc']:.5f},{valid['acc']:.5f},"
                f"{test['loss']:.5f},{test['auc']:.5f},{test['acc']:.5f},"
                f"{test['sen']:.5f},{test['spc']:.5f}\n"
            )
        if self.verbose:
            print(f"| Train ACC: {train['acc']:.5f} "
                  f"| Valid AUC: {valid['auc']:.5f} Valid ACC: {valid['acc']:.5f} "
                  f"| Test AUC: {test['auc']:.5f} Test ACC: {test['acc']:.5f} "
                  f"Test SEN: {test['sen']:.5f} Test SPC: {test['spc']:.5f}")

    def log_fold(self, iteration: int, fold: int, result: Dict[str, float]) -> None:
        with open(self.log_iter_path, 'a') as f:
            f.write(f"{fold + 1},{result['auc']:.5f},{result['acc']:.5f},"
                    f"{result['sen']:.5f},{result['spc']:.5f}\n")
        if self.verbose:
            print(f"----- Fold {fold + 1} / Iter {iteration + 1} | "
                  f"AUC: {result['auc']:.5f} ACC: {result['acc']:.5f} "
                  f"SEN: {result['sen']:.5f} SPC: {result['spc']:.5f}")

    def log_iter(self, iteration: int, results: List[Dict[str, float]]) -> None:
        with open_dict(self.cfg):
            self.cfg.n_complete = len(results)

        _mean = {k: np.mean([result[k] for result in results]) for k in results[0].keys()}
        _std = {k: np.std([result[k] for result in results]) for k in results[0].keys()}

        self.all_results.append(_mean)
        self.all_fold_results.append(results)

        with open(self.log_iter_path, 'a') as f:
            f.write(f'Avg.,{_mean["auc"]:.5f},{_mean["acc"]:.5f},{_mean["sen"]:.5f},{_mean["spc"]:.5f}\n')
            f.write(f'STD,{_std["auc"]:.5f},{_std["acc"]:.5f},{_std["sen"]:.5f},{_std["spc"]:.5f}\n')

        with open(self.log_filepath, 'a') as f:
            f.write(f'{iteration + 1},{_mean["auc"]:.5f},{_mean["acc"]:.5f},'
                    f'{_mean["sen"]:.5f},{_mean["spc"]:.5f}\n')

        if self.verbose:
            print(f">>>>> Iter {iteration + 1} | AUC: {_mean['auc']:.5f} ACC: {_mean['acc']:.5f} "
                  f"SEN: {_mean['sen']:.5f} SPC: {_mean['spc']:.5f}")

    def log_avg(self) -> None:
        _mean = {k: np.mean([result[k] for result in self.all_results]) for k in self.all_results[0].keys()}
        _std = {k: np.std([result[k] for result in self.all_results]) for k in self.all_results[0].keys()}

        with open(self.log_filepath, 'a') as f:
            f.write(f'Avg.,{_mean["auc"]:.5f},{_mean["acc"]:.5f},{_mean["sen"]:.5f},{_mean["spc"]:.5f}\n')
            f.write(f'STD,{_std["auc"]:.5f},{_std["acc"]:.5f},{_std["sen"]:.5f},{_std["spc"]:.5f}\n')

        if self.verbose:
            print(f">>>>>>>>>> Avg. AUC: {_mean['auc']:.5f} Avg. ACC: {_mean['acc']:.5f} "
                  f"Avg. SEN: {_mean['sen']:.5f} Avg. SPC: {_mean['spc']:.5f}")
