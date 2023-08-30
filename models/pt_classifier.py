from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from h5_pt_dataloader import ECGDataModule, HDF5ECGDataset
from collections import Counter
from time import perf_counter


class PatientCluster:
    def __init__(self, discriminator):
        self.__vectors = []
        self.__similarity_matrix = torch.zeros(size=(0, 0))
        self.__quality = torch.zeros(size=(0,))
        self.__consistency = 1.0
        self.__discriminator = discriminator
        self.__label = None

    def add_ecg_vector(self, vector, true_index, is_init):
        if self.__label is None:
            self.__label = true_index

        elem = (vector, true_index, is_init)
        self.__vectors.append(elem)
        n = len(self.__vectors)

        # Increase size of probability matrix by 1 row and 1 column
        new_similarity_matrix = torch.zeros(size=(n, n))
        new_similarity_matrix[:(n - 1), :(n - 1)] = self.__similarity_matrix

        if n == 1:
            self.__similarity_matrix = torch.ones(size=(1, 1))
            self.__quality = torch.ones(size=(1,))
        else:  # n >= 2
            # Calculate probs between new vector and old vectors
            old = self.get_patient_vectors()[:-1]
            a = torch.stack([vector for _ in old])
            b = torch.stack(old)
            probabilities = self.__discriminator(a, b)

            # Fill in the new row and column. Add 1 to diagonal.
            new_similarity_matrix[-1, :-1] = probabilities
            new_similarity_matrix[:-1, -1] = probabilities
            new_similarity_matrix[-1, -1] = 1.0

            self.__quality = torch.sum(new_similarity_matrix, dim=1)
            self.__consistency = torch.sum(new_similarity_matrix) / (n * (n - 1))
            self.__similarity_matrix = new_similarity_matrix

    def get_patient_vectors(self):
        return [v[0] for v in self.__vectors]

    def get_true_labels(self):
        return [v[1] for v in self.__vectors if not v[2]]

    def get_pred_label(self):
        return self.__label

    def get_quality(self):
        return self.__quality

    def get_consistency(self):
        return self.__consistency


class Database:
    def __init__(self, discriminator):
        self.__clusters: list[PatientCluster] = []
        self.__discriminator = discriminator
        self.__patient_id_to_cluster_id = {}

        # In our current setting, we never correct mistakes in the database unless they have been detected by the
        # discriminator. Fixed mistake counters below will always be 0. The code remains here in case this changes
        # in the future.
        self.__fix_creation = False
        self.__fix_mislabel_probability = 0.0

        self.__fixed_creation_mistakes = 0
        self.__fixed_non_creation_mistakes = 0
        self.__fixed_mislabel_mistakes = 0

        self.__backup: Database = None

    def save(self):
        self.__backup = deepcopy(self)

    def restore(self):
        self.__clusters = deepcopy(self.__backup.__clusters)
        self.__patient_id_to_cluster_id = deepcopy(self.__backup.__patient_id_to_cluster_id)

    def has_patient_id(self, patient_id):
        return patient_id in self.__patient_id_to_cluster_id

    def add_ecg_vector(self, vector, patient_id, cluster_id_guess: int = None):
        # Always find the true cluster ID the vector should belong to
        if patient_id not in self.__patient_id_to_cluster_id:
            true_cluster_id = -1
        else:
            true_cluster_id = self.__patient_id_to_cluster_id[patient_id]

        # In init phase: just store the vector in the database
        if cluster_id_guess is None:
            self.__store_vector(true_cluster_id, True, patient_id, vector)

        # In classification phase
        else:
            # If the classifier made a mistake...
            if cluster_id_guess != true_cluster_id:
                # ...and is trying to create a new cluster (for an existing patient)
                if cluster_id_guess == -1:
                    # We tolerate a state where multiple clusters have the same predicted label, which is always the
                    # patient whose vector was used to declare the cluster. However, creating new such clusters always
                    # counts as a mistake. If fix_creation is True, we automatically merge these clusters to leave the
                    # database in a less corrupted state.
                    self.__fixed_creation_mistakes += 1
                    if self.__fix_creation:
                        cluster_id_guess = true_cluster_id

                # ...and is simply mislabeling a signal
                else:
                    # The classifier is free to make mislabeling mistakes. If we fix any for the classifier, we count it
                    # as a mistake, otherwise the mistake will manifest itself as a corrupted database state.
                    if torch.rand(1) < self.__fix_mislabel_probability:
                        cluster_id_guess = true_cluster_id

                        if true_cluster_id == -1:
                            self.__fixed_non_creation_mistakes += 1
                        else:
                            self.__fixed_mislabel_mistakes += 1

            # Finally, store the vector in the database.
            self.__store_vector(cluster_id_guess, False, patient_id, vector)

        return true_cluster_id == -1

    def __store_vector(self, cluster_id_guess, is_init, patient_id, vector):
        # If the cluster should be created...
        if cluster_id_guess == -1:
            cluster = PatientCluster(self.__discriminator)
            self.__clusters.append(cluster)
            cluster.add_ecg_vector(vector, patient_id, is_init)
            self.__patient_id_to_cluster_id[patient_id] = len(self) - 1

        # ...or extended.
        else:
            cluster = self.__clusters[cluster_id_guess]
            cluster.add_ecg_vector(vector, patient_id, is_init)

    def __len__(self):
        return len(self.__clusters)

    def get_patient_cluster(self, index) -> PatientCluster:
        return self.__clusters[index]

    def get_predicted_and_true_labels(self):
        pred = []
        true = []
        for c in self.__clusters:
            pred_label = c.get_pred_label()
            for true_label in c.get_true_labels():
                true.append(true_label)
                pred.append(pred_label)
        return pred, true

    def get_prediction_stats(self):
        """
        :return: Number of correct positives and negatives, number of mislabel mistakes, number of creation mistakes, and
        total amount of classifications.
        """
        pred, true = self.get_predicted_and_true_labels()
        num_correct = sum([a == b for a, b in zip(pred, true)])
        return num_correct, self.__fixed_mislabel_mistakes, self.__fixed_non_creation_mistakes, \
            self.__fixed_creation_mistakes, len(true)

    def get_cluster_id_by_patient_id(self, patient_id):
        return self.__patient_id_to_cluster_id.get(patient_id, -1)


class Classifier:
    def __init__(self, embedding_model: torch.nn.Module, discriminator_model: torch.nn.Module, approach):
        self.__embedding = embedding_model
        self.__discriminator = discriminator_model
        self.__approach = approach

        self.__approach_handler = {
            'a1': self.__likelihood_approach_a1,
            'a2': self.__likelihood_approach_a2,
            'a3': self.__likelihood_approach_a3,
            'a4': self.__likelihood_approach_a4,
        }

        for param in self.__discriminator.parameters():
            param.requires_grad = False

        for param in self.__embedding.parameters():
            param.requires_grad = False

        self.__discriminator.train(False)
        self.__embedding.train(False)

    def make_generator(self, dataset, subset, mode, samples, seed, r):
        f, i_f = self.validate_dataset(dataset)

        if subset == 'whole':
            sample_size = samples
            ds = ECGDataModule(f'datasets/{dataset}', 1, mode, sample_size,
                               train_fraction=0,
                               dev_fraction=0,
                               test_fraction=1,
                               test_seed=r + seed)
            return iter(ds.test_dataloader())
        elif subset == 'dev':
            sample_size = samples * i_f[1]
            ds = ECGDataModule(f'datasets/{dataset}', 1, mode, sample_size,
                               train_fraction=f[0],
                               dev_fraction=f[1],
                               test_fraction=f[2],
                               dev_seed=r + seed)
            return iter(ds.val_dataloader())
        elif subset == 'test':
            sample_size = samples * i_f[2]
            ds = ECGDataModule(f'datasets/{dataset}', 1, mode, sample_size,
                               train_fraction=f[0],
                               dev_fraction=f[1],
                               test_fraction=f[2],
                               test_seed=r + seed)
            return iter(ds.test_dataloader())
        else:
            raise ValueError(f"Unknown subset {subset}.")

    def gallery_and_probe(self, dataset, subset, eval_samples, seed, repeats):
        records = []
        for r in tqdm(range(repeats), desc=f"Gallery and probe ({repeats=})"):
            generator = self.make_generator(dataset, subset, HDF5ECGDataset.Mode.MODE_GALLERY_PROBE, eval_samples,
                                            seed, r)

            all_pairs = []
            for el in tqdm(generator, desc="Embedding the signals...", total=eval_samples, leave=False):
                all_pairs.append(self.__embedding(el.transpose(-1, -2)))

            all_pairs = torch.stack(all_pairs)

            gallery = all_pairs[:, 0, ...]
            probe = all_pairs[:, 1, ...]

            correct = 0
            for i, p in enumerate(tqdm(probe, desc="Probing", leave=False)):
                a = torch.stack([p for _ in gallery])
                probabilities = self.__discriminator(a, gallery).reshape(-1)
                if torch.argmax(probabilities) == i:
                    correct += 1

            records.append({
                'accuracy': correct / len(probe),
                'samples': len(probe),
            })

        return records

    def evaluate(self, dataset: str, subset, sample_mode: str, init_patients: int, eval_samples: int,
                 overseer_mistake_search: list[float], decision_threshold: float,
                 likelihood_threshold: float, seed: int, repeats: int):

        if sample_mode == 'random':
            m = HDF5ECGDataset.Mode.MODE_ECG_WITH_ID_RANDOM
            sample_size = init_patients + eval_samples
        elif sample_mode == 'fill':
            m = HDF5ECGDataset.Mode.MODE_ECG_WITH_ID_FILL
            sample_size = init_patients
        else:
            raise ValueError(f"Unknown value {sample_mode}. Choose 'random' or 'fill'.")

        records = []
        for r in tqdm(range(repeats), desc=f"Overseer simulation ({repeats=})"):
            generator = self.make_generator(dataset, subset, m, sample_size, seed, r)
            database = Database(self.__discriminator)

            if sample_mode == 'fill':
                _, init_size = next(generator)
                _, rest_size = next(generator)
            else:
                init_size = init_patients
                rest_size = eval_samples

            # Database initialization
            for _ in tqdm(range(init_size + (rest_size - eval_samples)), desc="Building initial database...", leave=False):
                patient_id, vector = self.__fetch_patient(generator)
                database.add_ecg_vector(vector, patient_id)

            database.save()

            patient_ids = []
            vectors = []
            for _ in tqdm(range(eval_samples), desc="Loading eval samples", leave=False):
                patient_id, vector = self.__fetch_patient(generator)
                patient_ids.append(patient_id)
                vectors.append(vector)

            try:
                next(generator)
                raise AssertionError("Generator should not have any more samples...")
            except StopIteration:
                pass

            for om_rate in overseer_mistake_search:
                database.restore()

                stats = []
                # Classification process
                overseer_mistakes = int(eval_samples * om_rate)

                overseer_mistake_mask = np.zeros(eval_samples, dtype=bool)
                overseer_mistake_mask[np.random.choice(eval_samples, overseer_mistakes, replace=False)] = True

                for clinician_made_a_mistake, patient_id, vector in \
                        tqdm(zip(overseer_mistake_mask, patient_ids, vectors), total=eval_samples,
                             desc=f"Classifying patients (omr = {overseer_mistakes / eval_samples:.2f})", leave=False):

                    start_time_detect = perf_counter()
                    correct_cluster_id = database.get_cluster_id_by_patient_id(patient_id)
                    assert correct_cluster_id != -1
                    mistake_corrected = False
                    mistake_correction_hinted = False
                    mistake_detected = False

                    if clinician_made_a_mistake:
                        while True:
                            clinician_cluster_id = np.random.randint(0, len(database), size=1)[0]
                            if clinician_cluster_id != correct_cluster_id:
                                break
                    else:
                        clinician_cluster_id = correct_cluster_id

                    clinician_chosen_cluster = database.get_patient_cluster(clinician_cluster_id)
                    likelihood = self.__calculate_likelihood(vector, clinician_chosen_cluster)
                    if likelihood < likelihood_threshold:
                        mistake_detected = True

                    end_time_detect = perf_counter()

                    potential_cluster_ids, dec_pred, dec_true = \
                        self.__find_candidates(vector, database, decision_threshold, correct_cluster_id)
                    acc_dec = sum(sum(p == t for p, t in zip(P, T)) for P, T in zip(dec_pred, dec_true)) / sum(
                        len(P) for P in dec_pred)

                    start_time_suggest = perf_counter()

                    # Discriminator claims the overseer made a mistake
                    if mistake_detected:
                        if len(potential_cluster_ids) == 0:
                            # Discriminator claims the patient is not in the database.
                            best_fit = -1
                        else:
                            # Discriminator identifies a patient in the database as the true owner of the ECG
                            best_fit = max(potential_cluster_ids, key=lambda i:
                            self.__calculate_likelihood(vector, database.get_patient_cluster(i)))

                        if clinician_made_a_mistake:
                            if correct_cluster_id in potential_cluster_ids:
                                mistake_correction_hinted = True
                            if best_fit == correct_cluster_id:
                                mistake_corrected = True

                    end_time_suggest = perf_counter()

                    if clinician_made_a_mistake and mistake_detected:
                        database.add_ecg_vector(vector, patient_id, correct_cluster_id)
                    else:
                        database.add_ecg_vector(vector, patient_id, clinician_cluster_id)

                    clinician_cluster_decisions = dec_pred[clinician_cluster_id]
                    true_cluster_decisions = dec_pred[correct_cluster_id]
                    stats.append({
                        'clinician_mistake': clinician_made_a_mistake,
                        'mistake_detected': mistake_detected,
                        'mistake_corrected': mistake_corrected,
                        'mistake_correction_hinted': mistake_correction_hinted,
                        'accuracy_on_setup': acc_dec,
                        'OC_close_frac': sum(clinician_cluster_decisions) / len(clinician_cluster_decisions),
                        'true_close_frac': sum(true_cluster_decisions) / len(true_cluster_decisions),
                        'likelihood': likelihood,
                        'detect_time_taken_s': end_time_detect - start_time_detect,
                        'suggest_time_taken_s': end_time_suggest - start_time_suggest,
                    })

                # Evaluation process
                om_tp = sum(row['clinician_mistake'] and row['mistake_detected'] for row in stats)
                om_fp = sum(not row['clinician_mistake'] and row['mistake_detected'] for row in stats)
                om_fn = sum(row['clinician_mistake'] and not row['mistake_detected'] for row in stats)
                om_tn = sum(not row['clinician_mistake'] and not row['mistake_detected'] for row in stats)

                precision = om_tp / (om_tp + om_fp) if om_tp > 0 else 0
                recall = om_tp / (om_tp + om_fn) if om_tp > 0 else 0
                specificity = om_tn / (om_tn + om_fp) if om_tn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

                tp_tn, lm, ncm, cm, total = database.get_prediction_stats()
                om_total = sum(row['clinician_mistake'] for row in stats)
                # missed_total = sum(row['overseer_mistake'] and not row['mistake_detected'] for row in stats)

                p_at_r95, p_at_r95_threshold = self.precision_at_recall(overseer_mistakes, stats, target_recall=0.95)

                records.append({
                    'ecg_count': total,
                    'ecg_correct': tp_tn,
                    'patient_count': len(database),

                    'threshold': likelihood_threshold,
                    'OM_PRE': precision,
                    'OM_REC/SENS': recall,
                    'OM_SPEC': specificity,
                    'OM_F1': f1,
                    'OM_P@R95': p_at_r95,
                    'OM_P@R95_threshold': p_at_r95_threshold,

                    'OM_corrected_frac': (sum(row['clinician_mistake'] and row['mistake_corrected'] for row in
                                              stats) / om_total) if om_total > 0 else 0,
                    'OM_corr_hinted_frac': (sum(row['clinician_mistake'] and row['mistake_correction_hinted'] for row in
                                                stats) / om_total) if om_total > 0 else 0,

                    'overseer_mistakes': om_total,
                    'alarms': (om_tp + om_fp),
                    'false_alarms': om_fp,
                    'missed_alarms': om_fn,
                    'acc_dec': sum(row['accuracy_on_setup'] for row in stats) / len(stats),
                    'avg_detect_time_taken_s': sum(row['detect_time_taken_s'] for row in stats) / len(stats),
                    'avg_suggest_time_taken_s': sum(row['suggest_time_taken_s'] for row in stats) / len(stats),
                    # 'OM_close_frac': (sum(row['OC_close_frac'] and row['overseer_mistake'] for row in stats) / om_total) if om_total > 0 else 0,
                    # 'true_close_frac': sum(row['true_close_frac'] for row in stats) / len(stats),
                    # 'missed_close_frac': (sum(row['OC_close_frac'] and row['overseer_mistake'] and not row['mistake_detected'] for row in stats) / missed_total) if missed_total > 0 else 0,
                })

        return records

    @staticmethod
    def validate_dataset(dataset):
        fractions = {
            'c15p': (0.7, 0.1, 0.2),
            'ikem': (0.0, 0.5, 0.5),
            'ptbxl': (0.0, 0.5, 0.5),
            'ptb': (0.0, 0.0, 1.0),
        }
        inv_fractions = {
            'c15p': (None, 10, 5),
            'ikem': (None, 2, 2),
            'ptbxl': (None, 2, 2),
            'ptb': (None, None, 1),
        }
        try:
            f = fractions[dataset]
            i_f = inv_fractions[dataset]
        except KeyError:
            raise ValueError(f"Unknown dataset {dataset}. Choose one from {list(fractions.keys())}.")
        return f, i_f

    @staticmethod
    def precision_at_recall(overseer_mistakes, stats, target_recall):
        if overseer_mistakes == 0:
            return None, None

        assert 0 < target_recall < 1
        likelihoods_and_mistakes = [(1 - row['likelihood'], row['clinician_mistake']) for row in stats]
        current_recall = 0.0
        current_fp = 0
        p_at_r95 = None
        p_at_r95_threshold = None
        for lkh, mtk in sorted(likelihoods_and_mistakes, key=lambda x: x[0], reverse=True):
            if mtk == 1:
                current_recall += 1 / overseer_mistakes
                if current_recall >= target_recall:
                    p_at_r95_threshold = float(lkh)
                    p_at_r95 = overseer_mistakes * current_recall / (current_fp + overseer_mistakes * current_recall)
            else:
                current_fp += 1

        assert p_at_r95 is not None
        assert p_at_r95_threshold is not None

        return p_at_r95, p_at_r95_threshold

    def __fetch_patient(self, generator):
        ecg_signal, patient_id = next(generator)
        ecg_signal = ecg_signal.transpose(1, 2)
        vector = self.__embedding(ecg_signal).reshape(-1)
        patient_id = int(patient_id[0])
        return patient_id, vector

    def __find_candidates(self, vector, database: Database, decision_threshold: float,
                          correct_cluster_id: int = None):
        ret = []
        pred = []
        true = []

        for i in range(len(database)):
            cluster = database.get_patient_cluster(i)

            others = cluster.get_patient_vectors()
            a = torch.stack([vector for _ in others])
            b = torch.stack([other for other in others])
            probabilities = self.__discriminator(a, b).reshape(-1, 1)
            decisions = probabilities > decision_threshold
            if sum(decisions) >= 0.5 * len(decisions):
                ret.append(i)

            pred.append([int(decision) for decision in decisions])
            true.append([1] * len(decisions) if correct_cluster_id == i else [0] * len(decisions))

        if correct_cluster_id is not None:
            return ret, pred, true
        else:
            return ret

    def __calculate_likelihood(self, vector, cluster):
        try:
            return self.__approach_handler[self.__approach](vector, cluster)
        except KeyError as e:
            print(f"Unknown approach {self.__approach}.")
            raise e

    def __likelihood_approach_a1(self, vector, cluster: PatientCluster):
        others = cluster.get_patient_vectors()
        mean = torch.mean(torch.stack(others))
        return self.__discriminator(vector.reshape(1, -1), mean.reshape(1, -1))

    def __likelihood_approach_a2(self, vector, cluster: PatientCluster):
        others = cluster.get_patient_vectors()
        a = torch.stack([vector for _ in others])
        b = torch.stack([other for other in others])
        probabilities = self.__discriminator(a, b)
        return torch.mean(probabilities)

    def __likelihood_approach_a3(self, vector, cluster: PatientCluster):
        others = cluster.get_patient_vectors()
        a = torch.stack([vector for _ in others])
        b = torch.stack([other for other in others])
        probabilities = self.__discriminator(a, b)
        quality = cluster.get_quality()
        weighted_probabilities = probabilities * quality
        res = torch.sum(weighted_probabilities) / torch.sum(quality)
        return res

    def __likelihood_approach_a4(self, vector, cluster: PatientCluster):
        l_a3 = self.__likelihood_approach_a3(vector, cluster)

        return cluster.get_consistency() * l_a3


