# Code Review — Reinforcement Learning en Temps Continu avec Signatures

**Date :** 26 février 2026  
**Scope :** `src/` (agents, networks, environments, utils), entry points, configuration  
**Objectif du projet :** Implémenter et comparer l'Actor-Critic (CTAC) et le Value Gradient de Kenji Doya (2000) en temps continu, avec extension aux systèmes à retard via les signatures de chemins (path signatures).

---

## Table des matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Ce qui fonctionne bien](#2-ce-qui-fonctionne-bien)
3. [Problèmes critiques](#3-problèmes-critiques)
4. [Problèmes de conception / Design Smells](#4-problèmes-de-conception--design-smells)
5. [Fidélité à Doya (2000)](#5-fidélité-à-doya-2000)
6. [Correctness des algorithmes](#6-correctness-des-algorithmes)
7. [Qualité du code](#7-qualité-du-code)
8. [Recommendations prioritaires](#8-recommendations-prioritaires)

---

## 1. Vue d'ensemble de l'architecture

### Agents (hiérarchie d'héritage)

```
CTAC (base.py)              ← NumPy, Markov, linéaire, LQR uniquement
CTACSignature (signatures.py) ← NumPy, signatures, systèmes à retard
CTACSignatureJAX (signatures_jax.py) ← JAX, signatures, Flax networks
    ├── CTACJAX (base_jax.py)  ← hérite de CTACSignatureJAX mais n'utilise PAS les signatures (Markov)
    └── (utilisé directement pour signatures + JAX)
ContinuousValueGradient (value_gradient_jax.py) ← Value Gradient de Doya, JAX, signatures
CSAC (CSAC_jax.py)          ← Continuous Soft Actor-Critic (Han), JAX, signatures, replay buffer
```

### Environnements

```
Environment (env_rk.py)         ← NumPy, RK4, DDE linéaires
JAXDDEEnv (env_rk_jax.py)      ← JAX, RK4, DDE linéaires, jit-compilé
    ├── ChemicalReactionEnv     ← Non-linéaire, 4D, retard
    └── MackeyGlass1DEnv        ← Non-linéaire, 1D, retard
JAXEnvWrapper                   ← Wrapper avec historique, step/reset
```

### Signatures

```
SlidingSignature (dynamic_signature.py)     ← NumPy, utilise signax
SlidingSignatureJAX (dynamic_signature.py)  ← JAX, JIT-compilé, buffer circulaire
```

---

## 2. Ce qui fonctionne bien

### 2.1 Architecture modulaire avec flags
La classe de base `CTACSignatureJAX` utilise des flags (`discounted`, `semi_gradient`, `integral_td`, `actor_oracle`, `critic_oracle`) plutôt que des sous-classes pour chaque variante. L'approche est bonne : elle réduit l'explosion combinatoire de classes et permet de tester facilement toutes les combinaisons de l'article de Doya.

### 2.2 Implémentation JAX bien pensée
- Les fonctions de mise à jour (critic, actor) sont correctement JIT-compilées via des closures (`_make_critic_update_fn`, `_make_actor_update_fn`).
- Les accumulations de métriques sont faites avec des `jnp.array` pour éviter les synchronisations CPU/GPU inutiles, avec conversion `float()` seulement en fin d'épisode.
- Le pattern « *dirty flag* » pour la signature (`_signature_dirty`) évite les recalculs inutiles.

### 2.3 Environnement DDE avec RK4
- L'intégration RK4 pour les DDE est correctement implémentée avec interpolation des états retardés.
- La séparation `JAXDDEEnv` (pure, JIT-compilable) / `JAXEnvWrapper` (stateful, historique) est propre.
- L'utilisation de `jax.lax.scan` dans `step` pour les sous-pas est idiomatique JAX.

### 2.4 Signatures de chemins
- L'utilisation de `signax` (bibliothèque JAX pour les signatures) est appropriée.
- Les augmentations (temps, origine, biais) sont correctement implémentées.
- La taille de fenêtre adaptée au retard ($\text{window} = \lceil\tau / \Delta t\rceil + 1$) est bien motivée théoriquement.

### 2.5 Bruit lisse (Gaussian Process)
Le pré-échantillonnage du bruit d'exploration via un processus gaussien (noyau RBF) est une bonne idée pour les systèmes en temps continu — cela fournit une exploration temporellement corrélée plutôt qu'un bruit i.i.d. à chaque pas.

### 2.6 Checkpointing et early stopping
L'évaluation périodique sans bruit (`_evaluate_noiseless`), la sauvegarde du meilleur checkpoint, et l'early stopping par patience sont de bonnes pratiques.

### 2.7 Infrastructure Hydra + W&B
L'intégration Hydra pour la configuration et W&B pour le logging est professionnelle et facilite les expériences à grande échelle (multirun).

---

## 3. Problèmes critiques

### 3.1 ⚠️ Hiérarchie d'héritage inversée : `CTACJAX` hérite de `CTACSignatureJAX`

**Fichier :** `base_jax.py` L47

```python
class CTACJAX(CTACSignatureJAX):
```

`CTACJAX` est censé être l'agent Markov (sans signatures), mais il **hérite** de `CTACSignatureJAX` (l'agent avec signatures). Il override ensuite toutes les fonctions JIT pour utiliser l'état brut au lieu de la signature. C'est une inversion de la relation IS-A :
- ✅ `CTACSignatureJAX` IS-A extension de CTAC avec signatures
- ❌ `CTACJAX` IS-A `CTACSignatureJAX` — faux, c'est un sous-ensemble

**Conséquences :**
- `CTACJAX` crée quand même un `SlidingSignatureJAX` et ses buffers inutilement.
- Les JIT functions sont compilées deux fois (parent puis override).
- Des attributs de signature traînent sans être utilisés, source de confusion.

**Recommandation :** Extraire une classe de base abstraite commune (`BaseCTACJAX`) dont hériteraient séparément `CTACSignatureJAX` et `CTACJAX`.

### 3.2 ⚠️ Code dupliqué entre `base_jax.py` et `base.py` dans `_train_step`

`CTACJAX._train_step` (L226-300) a un bloc de code dupliqué pour le `delayed_state` :

```python
if self.delayed_state:
    if x_t.shape[0] == self.env.N:  # state not augmented!
        x_t = jnp.concatenate([x_t, self.wrapper.current_delayed_state], axis=0)
        x_scaled = x_t / self.scale
```

Ce bloc apparaît **deux fois consécutivement** (L238-241, puis L244-247). La deuxième occurrence est dead code ou un bug de copier-coller.

### 3.3 ⚠️ Duplication du bruit GP (`_on_episode_start`)

Le code de génération du bruit GP (Cholesky + sampling) est **copié-collé identiquement** dans :
- `signatures_jax.py` L641-680
- `value_gradient_jax.py` L310-348

Ce sont ~40 lignes identiques. Toute correction dans un fichier doit être reproduite dans l'autre.

### 3.4 ⚠️ Duplication de `_evaluate_noiseless`

La méthode `_evaluate_noiseless` est dupliquée (avec des variantes subtiles) dans :
- `signatures_jax.py` L790-850 (sauvegarde/restauration du state)
- `base_jax.py` L363-425 (idem, avec `delayed_state`)
- `value_gradient_jax.py` L580-650 (avec `_path_data_dirty`)

Les sections commentées dans `value_gradient_jax.py` (`for _ in range(self.burning_steps)...`) montrent une incertitude sur la bonne implémentation — signe que le code devrait être factorisé.

### 3.5 ⚠️ Scaling dt incohérent dans les optimizers

Dans `signatures_jax.py` L551-557, les updates sont multipliés par `dt` **après** l'optimizer :

```python
updates = jax.tree_util.tree_map(lambda u: u * dt, updates)
```

Mais dans `base_jax.py` L179-181, les **gradients** sont multipliés par `dt` **avant** l'optimizer :

```python
grads = jax.tree_util.tree_map(lambda g: g * dt, grads)
```

Ces deux approches ne sont PAS équivalentes car Adam normalise par la variance du gradient. Multiplier le gradient par `dt` avant Adam change le rapport signal-sur-bruit interne à Adam. Multiplier l'update par `dt` après Adam ne change que l'amplitude du pas. L'approche `grads * dt` de `base_jax.py` est probablement incorrecte pour Adam (elle introduit un biais dans les moments).

Pour le value gradient (`value_gradient_jax.py`), le scaling `dt` n'est pas appliqué du tout aux updates, ce qui est aussi incohérent.

**Recommandation :** Choisir une seule convention. L'approche la plus propre est d'absorber `dt` dans le learning rate : `lr_effective = lr * dt`.

### 3.6 ⚠️ Mélange `np.random` et `jax.random` dans le GP

Dans `_on_episode_start` (signatures_jax.py L660) :

```python
white_noise = np.random.randn(n_points, action_dim)
```

Ceci utilise le RNG global NumPy, **non reproductible** et non contrôlé par `self.key`. Dans un contexte JAX où la reproductibilité est gérée par `jax.random.PRNGKey`, c'est une fuite de stochasticité.

---

## 4. Problèmes de conception / Design Smells

### 4.1 Prolifération de paramètres dans `__init__`

`CTACSignatureJAX.__init__` a **30+ paramètres** (incluant `smooth_noise`, `noise_length_scale`, `x0`, `burning_steps`, `eval_callback`, `state_augmentation`, `origin_augmentation`, `time_origin`...). C'est un symptôme de « God Class ».

**Recommandation :** Grouper les paramètres en dataclasses :
- `NoiseConfig(sigma, schedule, smooth, length_scale)`
- `SignatureConfig(depth, window_size, time_augmentation, origin_augmentation, bias)`
- `TrainingConfig(n_episodes, max_time, lr_actor, lr_critic, ...)`

### 4.2 `value_gradient_jax_markov.py` — fichier quasi-vide

```python
from value_gradient_jax import ValueGradientAgent
```

Ce fichier d'une seule ligne importe une classe qui n'existe pas (`ValueGradientAgent` — la vraie classe s'appelle `ContinuousValueGradient`). C'est dead code / un placeholder abandonné.

### 4.3 Confusion cost / reward dans les commentaires et variables

Le projet utilise des **coûts** (négatifs = mauvais) mais les variables s'appellent tantôt `reward` tantôt `cost`, parfois interchangeablement :

- `value_gradient_jax.py` L542 : `#CAREFUL, cost is in fact reward here, so higher is better`
- `step_metrics.reward` contient ` r * dt` où `r < 0` (c'est un coût)
- `episode_metrics['cost']` accumule `step_metrics.reward` (le nommage est inversé)

**Recommandation :** Standardiser une convention (par ex. toujours `cost >= 0`, `reward = -cost`) et renommer les variables.

### 4.4 `base.py` et `signatures.py` — implémentations NumPy orphelines

Ces fichiers (NumPy, non-JAX) ne sont plus utilisés par `main_unified.py` ni `src/training/train.py`. Ils servent uniquement via `train.py` (legacy). Le registre (`registry.py`) ne référence que `base.py`.

L'existence parallèle de deux implémentations (NumPy et JAX) pour le même algorithme est source de confusion. Les corrections ne sont appliquées que dans la version JAX.

### 4.5 `CriticFlax` dupliqué dans deux fichiers

`CriticFlax` et `CriticFlaxLayerNorm` sont définis **identiquement** dans :
- `src/networks/LQR_actor_critics.py`
- `src/networks/value_gradient_nets.py`

Le `value_gradient_jax.py` importe depuis `value_gradient_nets.py` alors que les mêmes classes existent déjà dans `LQR_actor_critics.py`.

### 4.6 `# type: ignore` omniprésent

Le code contient **~80+** commentaires `# type: ignore`. Cela masque des vrais problèmes de typage, notamment les attributions dynamiques comme `ctx.reward = reward` sur un `@dataclass` qui a `reward: float = 0.0` — techniquement OK mais le `# type: ignore` est superflu et cache d'éventuels vrais problèmes.

### 4.7 Pas de tests unitaires dans `test/`

Le dossier `test/` existe mais n'a pas été peuplé (aucun fichier `.py` visible). Pour des algorithmes aussi sensibles mathématiquement, l'absence de tests est un risque majeur.

---

## 5. Fidélité à Doya (2000)

### 5.1 Actor-Critic en temps continu (CTAC)

L'article de Doya définit :

**TD error (continuous) :**
$$\delta(t) = r(t) + \frac{\partial V}{\partial t} - \frac{V(t)}{\tau}$$

**Critic update :**
$$\dot{w}_c = \eta_c \cdot \delta(t) \cdot \nabla_w V$$

**Actor update (stochastic policy gradient) :**
$$\dot{w}_a = \eta_a \cdot \delta(t) \cdot \frac{\partial \ln p(u | x)}{\partial w_a}$$

#### ✅ Ce qui est fidèle :

- La forme de l'erreur TD est correcte dans les variants `discounted=True` et `discounted=False` :
  ```python
  td = ctx.reward + ctx.V_dot  # undiscounted
  td -= ctx.V_t / self.tau     # discounted
  ```
  
- Le gradient de l'acteur pour une politique gaussienne $p(u|x) = \mathcal{N}(\mu(x), \sigma^2)$ :
  ```python
  grad_log_policy = noise / (sigma ** 2)
  loss = -td_error * jnp.sum(grad_log_policy * mu)
  ```
  C'est correct : $\nabla_w \ln p = \frac{(u - \mu)}{\sigma^2} \cdot \nabla_w \mu = \frac{\epsilon}{\sigma^2} \cdot \nabla_w \mu$ où $\epsilon = u - \mu$ est le bruit.

- Le scaling par `dt` pour la discrétisation du temps continu est présent (bien que parfois incohérent, cf. §3.5).

#### ⚠️ Points d'attention :

1. **Le critic utilise un semi-gradient par défaut** (`stop_gradient` sur `V_next`) dans les versions JAX, ce qui correspond au cas `semi_gradient=True`. Doya utilise un gradient complet dans l'article. Le flag existe mais il faudrait vérifier qu'il est bien testé avec `semi_gradient=False`.

2. **Le `loss = 0.5 * td_error^2`** utilisé dans les versions JAX est le squared TD error minimisé par SGD. Ce n'est pas exactement la règle de Doya ($\dot{w} = \eta \cdot \delta \cdot \nabla V$) mais c'est mathématiquement équivalent dans le cas semi-gradient puisque $\nabla_w (0.5 \delta^2) = \delta \cdot \nabla_w V$ quand on stop-gradient le target.

3. **Version intégrale :** L'option `integral_td` ($\delta = r \cdot dt + \Delta V$) est une discrétisation alternative correcte.

### 5.2 Value Gradient (Doya §3.3)

L'article de Doya propose une méthode alternative :

**Greedy policy (model-based) :**
$$u^*(t) = \arg\max_u \left[ r(x, u) + \frac{\partial V}{\partial x} \cdot F(x, u) \right]$$

Pour le cas LQR avec $r = -x^T Q x - u^T R u$ et $F = Ax + Bu$ :
$$u^* = \frac{1}{2} R^{-1} B^T \frac{\partial V}{\partial x}$$

#### ✅ Ce qui est fidèle :

L'implémentation dans `value_gradient_jax.py` :
```python
def select_action(critic_params, path_data, R, x_current):
    B = get_B_fn(x_current)
    grad_path = grad_V_fn(critic_params, path_data)
    end_gradient = grad_path[-1]
    u = 1/2 * jnp.linalg.inv(R) @ B.T @ end_gradient
```

C'est la formule correcte de Doya pour le value gradient dans le cas continu. Le gradient est pris par rapport au **chemin** (path), ce qui est l'extension signature de la formule originale $\partial V / \partial x$.

#### ⚠️ Points d'attention :

1. **`end_gradient = grad_path[-1]`** : Le gradient est pris par rapport au chemin complet, puis seule la dernière composante (correspondant à l'état courant dans le chemin) est utilisée. Cela fonctionne si la signature est construite à partir du chemin et que le gradient de la signature par rapport au dernier point du chemin est ce qu'on veut. Cependant, c'est une approximation — le vrai gradient $\partial V / \partial x$ dans l'espace des signatures n'est pas exactement le gradient par rapport au dernier point du chemin brut.

2. **Target network :** Doya n'utilise pas de target network. L'ajout (`target_params`, Polyak averaging) est emprunté au DRL moderne (DDPG/TD3) et n'est pas dans l'article original. C'est justifiable pour la stabilité mais c'est une déviation.

3. **Pas d'acteur dans le value gradient :** C'est correct — la méthode value gradient est model-based et n'a pas besoin d'acteur séparé. L'action est dérivée directement du gradient de la valeur. Le code reflète bien cela.

4. **Reward computation dans `step` :** Pour le value gradient sur les systèmes non-linéaires (Chemical, Mackey-Glass), le calcul $u = R^{-1} B^T \nabla_x V / 2$ suppose une forme quadratique du coût en $u$, ce qui est vérifié ($r = -u^T R u + ...$). C'est correct.

---

## 6. Correctness des algorithmes

### 6.1 Gradient clipping par élément vs par norme

```python
grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -10.0, 10.0), grads)
```

Le clipping **par élément** change la direction du gradient. Pour l'optimisation, le clipping **par norme globale** (`optax.clip_by_global_norm`) est préférable car il préserve la direction.

Le CSAC utilise correctement `optax.clip_by_global_norm` dans sa chaîne d'optimizer, mais les autres agents utilisent le clipping par élément.

### 6.2 Problème de normalisation de la signature

La signature d'un chemin n'est pas invariante par mise à l'échelle : $\text{Sig}(\lambda \cdot X) = \lambda^{level} \cdot \text{Sig}(X)$ au niveau $k$. Quand le scaling `self.scale` change l'amplitude des entrées, les composantes de niveau 1 et 2 de la signature auront des ordres de grandeur très différents.

L'option `normalize_sigs` (avec `LayerNorm`) existe mais son effet sur la convergence n'est pas systématiquement évalué.

### 6.3 Adam avec $\beta_1 = 0.1$ pour l'acteur

```python
self.actor_optimizer = optax.adam(self.actor_lr, b1=0.1)
```

Ce choix (typiquement $\beta_1 = 0.9$) est inhabituel. Un $\beta_1$ faible donne un moment qui oublie très vite, ce qui est proche de RMSProp. Ce n'est pas nécessairement faux mais devrait être documenté/justifié.

### 6.4 Le preheat utilise `action = jnp.zeros(...)` 

Pendant la phase de preheat, l'action est toujours zéro :
```python
action = jnp.zeros(self.env.B.shape[1])
```

Pour les systèmes avec un point d'équilibre non-trivial (Mackey-Glass à $x^* = 0.5$), cela peut initialiser le buffer de signatures dans une région non représentative. Utiliser la politique actuelle (même non-entraînée) serait plus cohérent.

### 6.5 `_is_episode_done` — signature différente selon les agents

- `base.py` et `signatures.py` : `_is_episode_done(x, t)` — prend le temps en argument
- `signatures_jax.py` et `value_gradient_jax.py` : `_is_episode_done(x)` — lit le temps depuis `self.wrapper.state.t`

Cette incohérence n'est pas un bug (chaque agent utilise sa propre version) mais complique la maintenance.

---

## 7. Qualité du code

### 7.1 Dead code et commentaires obsolètes

- Nombreux blocs de `print` et `jax.debug.print` commentés dans CSAC (`CSAC_jax.py` L320-340).
- `_compute_reward` est marqué `#no more used` mais n'est pas supprimé.
- `env_params` est stocké dans `__init__` mais rarement utilisé (sauf pour `save()`).
- `value_gradient_jax_markov.py` est un fichier mort avec un import cassé.
- Sections `# =============================================================================` de test à la fin de chaque fichier agent — devraient être dans `test/` ou des scripts séparés.

### 7.2 Conventions de nommage

| Variable | Signification | Problème |
|----------|--------------|----------|
| `self.tau` | Discount factor $\tau$ (Doya) | Confondu avec `tau_polyak` dans le value gradient |
| `ctx.reward` | En fait un coût négatif | Nommage inversé |
| `episode_metrics['cost']` | Accumulation de `step_metrics.reward` | Incohérent |
| `self.scale` | Scaling des états | Parfois `divergence_threshold`, parfois un param |
| `x_scaled` | $x / \text{scale}$ | Mais les signatures sont calculées sur `x_scaled`, pas `x` — est-ce voulu ? |

### 7.3 Gestion mémoire

La suppression périodique des données (`wrapper._data.clear()`) est nécessaire pour les longs épisodes, mais elle est faite de façon ad-hoc avec `memory_clear_interval`. Il serait plus propre de ne pas stocker les données si elles ne sont pas nécessaires pendant l'entraînement (un flag `store_history=False` sur le wrapper).

### 7.4 Reproductibilité

- Le `rng_key` est bien propagé dans JAX.
- MAIS : le seed est parfois hardcodé (`rng_key=42` par défaut) et le GP utilise `np.random.randn` (cf. §3.6).
- Il n'y a aucun mécanisme pour logger le seed utilisé dans les métriques.

---

## 8. Recommendations prioritaires

### 🔴 Priorité haute (bugs/correctness)

| # | Action | Fichiers |
|---|--------|----------|
| 1 | **Fixer le code dupliqué `delayed_state`** dans `_train_step` | `base_jax.py` L238-247 |
| 2 | **Harmoniser le scaling `dt`** (choisir : `lr_eff = lr * dt` ou `update * dt`, pas les deux) | `signatures_jax.py`, `base_jax.py` |
| 3 | **Remplacer `np.random.randn`** par `jax.random` dans le bruit GP | `signatures_jax.py`, `value_gradient_jax.py` |
| 4 | **Supprimer `value_gradient_jax_markov.py`** (import cassé) | `value_gradient_jax_markov.py` |
| 5 | **Clipping par norme globale** au lieu de per-element pour tous les agents | Tous les agents JAX |

### 🟡 Priorité moyenne (design/maintenabilité)

| # | Action | Fichiers |
|---|--------|----------|
| 6 | **Refactorer la hiérarchie** : extraire `BaseCTACJAX` → `CTACSignatureJAX` / `CTACJAX` | `base_jax.py`, `signatures_jax.py` |
| 7 | **Factoriser le bruit GP** dans une classe `GPNoise` réutilisable | `signatures_jax.py`, `value_gradient_jax.py` |
| 8 | **Factoriser `_evaluate_noiseless`** dans une méthode commune ou un mixin | Tous les agents JAX |
| 9 | **Unifier `CriticFlax`** — supprimer le doublon dans `value_gradient_nets.py` | `networks/` |
| 10 | **Standardiser la convention cost/reward** dans les noms de variables et métriques | Global |
| 11 | **Grouper les paramètres `__init__`** en dataclasses (`NoiseConfig`, `SignatureConfig`, etc.) | `signatures_jax.py` |

### 🟢 Priorité basse (qualité/polish)

| # | Action | Fichiers |
|---|--------|----------|
| 12 | **Ajouter des tests unitaires** (au minimum : TD error, gradient correctness, rollout) | `test/` |
| 13 | **Supprimer le dead code** (blocs commentés, `_compute_reward` non utilisé, etc.) | Global |
| 14 | **Archiver les agents NumPy** (`base.py`, `signatures.py`) ou les marquer comme legacy | `src/agents/` |
| 15 | **Documenter le choix $\beta_1 = 0.1$** pour Adam dans l'acteur | `signatures_jax.py` |
| 16 | **Logger le seed** dans les métriques pour la reproductibilité | `train.py` |

---

## Annexe : Résumé des fichiers

| Fichier | Lignes | Rôle | Status |
|---------|--------|------|--------|
| `agents/base.py` | 591 | CTAC NumPy, Markov | Legacy, non utilisé par `main_unified` |
| `agents/signatures.py` | 534 | CTAC + Signatures NumPy | Legacy |
| `agents/signatures_jax.py` | 1055 | CTAC + Signatures JAX | **Principal** — agent signature AC |
| `agents/base_jax.py` | 484 | CTAC JAX Markov (hérite signature) | Baseline Markov |
| `agents/value_gradient_jax.py` | 672 | Value Gradient Doya + Signatures | **Principal** — value gradient |
| `agents/CSAC_jax.py` | 553 | Continuous SAC | Comparaison |
| `agents/value_gradient_jax_markov.py` | 1 | Import cassé | **À supprimer** |
| `agents/registry.py` | 72 | Factory legacy | Inutilisé par `main_unified` |
| `networks/LQR_actor_critics.py` | 200 | Réseaux NumPy + Flax | OK |
| `networks/value_gradient_nets.py` | 36 | Doublon de CriticFlax | **À supprimer** |
| `utils/dynamic_signature.py` | 229 | Signatures glissantes | Bien conçu |
| `utils/step_context.py` | 85 | Conteneurs de données | OK |
| `utils/step_metrics.py` | 12 | Métriques par pas | OK |
| `utils/experience_replay_buffer.py` | 50 | Replay buffer pour CSAC | OK |
| `env_rk.py` | 215 | Env DDE NumPy | Legacy |
| `env_rk_jax.py` | 150 | Env DDE JAX | **Principal** |
| `chemical_process.py` | 186 | Env non-linéaire 4D | OK |
| `mackey_glass_1D.py` | 226 | Env Mackey-Glass 1D | OK |
| `training/train.py` | 216 | Training unifié Hydra | OK |
| `main_unified.py` | 204 | Point d'entrée + W&B | OK |
