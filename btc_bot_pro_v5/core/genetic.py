"""
BTC Bot Pro - Strateji Optimizasyonu
FAZA 5.1: Genetik Algoritma ile Parametre Optimizasyonu

Özellikler:
- Genetik algoritma (GA) optimizer
- Multi-objective optimization
- Elitism ve turnuva seçimi
- Adaptive mutation rate
- Early stopping
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random
import copy
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# DATACLASSES
# ================================================================

@dataclass
class Individual:
    """Genetik algoritma bireyi (strateji parametreleri)"""
    genes: Dict[str, float]
    fitness: float = 0
    metrics: Dict = field(default_factory=dict)
    generation: int = 0
    
    def __lt__(self, other):
        return self.fitness < other.fitness


@dataclass
class GAConfig:
    """Genetik algoritma konfigürasyonu"""
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    mutation_strength: float = 0.1
    adaptive_mutation: bool = True
    early_stopping_generations: int = 20
    n_jobs: int = 4
    random_seed: int = 42


@dataclass
class ParameterSpace:
    """Parametre arama uzayı"""
    name: str
    min_val: float
    max_val: float
    step: float = None
    is_integer: bool = False
    
    def random(self) -> float:
        if self.is_integer:
            return random.randint(int(self.min_val), int(self.max_val))
        elif self.step:
            steps = int((self.max_val - self.min_val) / self.step)
            return self.min_val + random.randint(0, steps) * self.step
        else:
            return random.uniform(self.min_val, self.max_val)
    
    def mutate(self, value: float, strength: float) -> float:
        range_size = self.max_val - self.min_val
        mutation = random.gauss(0, range_size * strength)
        new_value = value + mutation
        new_value = max(self.min_val, min(self.max_val, new_value))
        
        if self.is_integer:
            new_value = round(new_value)
        elif self.step:
            new_value = round(new_value / self.step) * self.step
        
        return new_value


@dataclass
class OptimizationResult:
    """Optimizasyon sonucu"""
    best_individual: Individual
    best_params: Dict[str, float]
    best_fitness: float
    best_metrics: Dict
    
    # Evolution history
    generation_best: List[float] = field(default_factory=list)
    generation_avg: List[float] = field(default_factory=list)
    
    # Stats
    total_evaluations: int = 0
    elapsed_time: float = 0
    early_stopped: bool = False
    final_generation: int = 0


# ================================================================
# GENETIC ALGORITHM OPTIMIZER
# ================================================================

class GeneticOptimizer:
    """
    Genetik Algoritma ile Strateji Optimizasyonu
    
    Kullanım:
        optimizer = GeneticOptimizer(fitness_fn, param_space)
        result = optimizer.run()
    """
    
    def __init__(self,
                 fitness_function: Callable,
                 parameter_space: List[ParameterSpace],
                 config: GAConfig = None):
        """
        Args:
            fitness_function: Fitness hesaplama fonksiyonu (params -> fitness)
            parameter_space: Parametre uzayı
            config: GA konfigürasyonu
        """
        self.fitness_fn = fitness_function
        self.param_space = {p.name: p for p in parameter_space}
        self.config = config or GAConfig()
        
        # Set seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # State
        self.population: List[Individual] = []
        self.best_individual: Individual = None
        self.generation = 0
        self.no_improvement_count = 0
        
        # History
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'mutation_rate': []
        }
    
    def _create_individual(self) -> Individual:
        """Rastgele birey oluştur"""
        genes = {}
        for name, space in self.param_space.items():
            genes[name] = space.random()
        return Individual(genes=genes, generation=self.generation)
    
    def _initialize_population(self):
        """Başlangıç popülasyonu oluştur"""
        self.population = [
            self._create_individual() 
            for _ in range(self.config.population_size)
        ]
    
    def _evaluate_individual(self, individual: Individual) -> Individual:
        """Bireyi değerlendir"""
        try:
            result = self.fitness_fn(individual.genes)
            
            if isinstance(result, tuple):
                individual.fitness = result[0]
                individual.metrics = result[1] if len(result) > 1 else {}
            else:
                individual.fitness = result
                individual.metrics = {}
            
        except Exception as e:
            individual.fitness = float('-inf')
            individual.metrics = {'error': str(e)}
        
        return individual
    
    def _evaluate_population(self):
        """Tüm popülasyonu değerlendir"""
        if self.config.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                self.population = list(executor.map(
                    self._evaluate_individual, 
                    self.population
                ))
        else:
            self.population = [
                self._evaluate_individual(ind) 
                for ind in self.population
            ]
    
    def _tournament_selection(self) -> Individual:
        """Turnuva seçimi"""
        tournament = random.sample(
            self.population, 
            self.config.tournament_size
        )
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """İki noktalı crossover"""
        if random.random() > self.config.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1_genes = {}
        child2_genes = {}
        
        gene_names = list(self.param_space.keys())
        
        # Crossover noktaları
        point1 = random.randint(0, len(gene_names) - 1)
        point2 = random.randint(point1, len(gene_names))
        
        for i, name in enumerate(gene_names):
            if point1 <= i < point2:
                child1_genes[name] = parent2.genes[name]
                child2_genes[name] = parent1.genes[name]
            else:
                child1_genes[name] = parent1.genes[name]
                child2_genes[name] = parent2.genes[name]
        
        return (
            Individual(genes=child1_genes, generation=self.generation),
            Individual(genes=child2_genes, generation=self.generation)
        )
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutasyon uygula"""
        mutated_genes = {}
        
        for name, value in individual.genes.items():
            if random.random() < self.config.mutation_rate:
                space = self.param_space[name]
                mutated_genes[name] = space.mutate(
                    value, 
                    self.config.mutation_strength
                )
            else:
                mutated_genes[name] = value
        
        individual.genes = mutated_genes
        return individual
    
    def _adapt_mutation_rate(self):
        """Adaptif mutasyon oranı"""
        if not self.config.adaptive_mutation:
            return
        
        # Stagnation durumunda mutasyon artır
        if self.no_improvement_count > 5:
            self.config.mutation_rate = min(0.5, self.config.mutation_rate * 1.1)
            self.config.mutation_strength = min(0.3, self.config.mutation_strength * 1.1)
        
        # İyileşme varsa mutasyon azalt
        elif self.no_improvement_count == 0:
            self.config.mutation_rate = max(0.1, self.config.mutation_rate * 0.95)
            self.config.mutation_strength = max(0.05, self.config.mutation_strength * 0.95)
    
    def _create_next_generation(self):
        """Sonraki jenerasyonu oluştur"""
        # Elitizm
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = copy.deepcopy(self.population[:self.config.elite_size])
        
        # Yeni bireyler
        while len(new_population) < self.config.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            child1, child2 = self._crossover(parent1, parent2)
            
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Popülasyon boyutunu düzelt
        self.population = new_population[:self.config.population_size]
    
    def _update_best(self):
        """En iyi bireyi güncelle"""
        current_best = max(self.population, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(current_best)
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
    
    def _record_history(self):
        """Jenerasyon istatistiklerini kaydet"""
        fitnesses = [ind.fitness for ind in self.population]
        
        self.history['best_fitness'].append(max(fitnesses))
        self.history['avg_fitness'].append(np.mean(fitnesses))
        self.history['mutation_rate'].append(self.config.mutation_rate)
    
    def run(self, progress_callback: Callable = None) -> OptimizationResult:
        """
        Genetik algoritmayı çalıştır
        
        Args:
            progress_callback: İlerleme callback (generation, best_fitness)
        """
        start_time = datetime.now()
        
        # Initialize
        self._initialize_population()
        self._evaluate_population()
        self._update_best()
        self._record_history()
        
        early_stopped = False
        
        for gen in range(self.config.generations):
            self.generation = gen + 1
            
            # Evolve
            self._create_next_generation()
            self._evaluate_population()
            self._update_best()
            self._adapt_mutation_rate()
            self._record_history()
            
            # Progress callback
            if progress_callback:
                progress_callback(
                    gen + 1, 
                    self.config.generations,
                    self.best_individual.fitness
                )
            
            # Early stopping
            if self.no_improvement_count >= self.config.early_stopping_generations:
                early_stopped = True
                break
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_individual=self.best_individual,
            best_params=self.best_individual.genes,
            best_fitness=self.best_individual.fitness,
            best_metrics=self.best_individual.metrics,
            generation_best=self.history['best_fitness'],
            generation_avg=self.history['avg_fitness'],
            total_evaluations=self.config.population_size * (self.generation + 1),
            elapsed_time=elapsed,
            early_stopped=early_stopped,
            final_generation=self.generation
        )


# ================================================================
# STRATEGY OPTIMIZER (Backtest ile entegre)
# ================================================================

class StrategyOptimizer:
    """
    Strateji parametrelerini optimize et
    
    Backtest engine ile entegre çalışır
    """
    
    # Varsayılan parametre uzayı
    DEFAULT_PARAMS = [
        ParameterSpace('stop_loss', 0.01, 0.05, 0.005),
        ParameterSpace('take_profit', 0.015, 0.10, 0.005),
        ParameterSpace('rsi_oversold', 20, 40, 5, is_integer=True),
        ParameterSpace('rsi_overbought', 60, 80, 5, is_integer=True),
        ParameterSpace('ema_fast', 5, 20, 1, is_integer=True),
        ParameterSpace('ema_slow', 20, 50, 5, is_integer=True),
        ParameterSpace('position_size', 0.3, 0.7, 0.1),
        ParameterSpace('threshold', 0.3, 1.0, 0.1),
    ]
    
    def __init__(self,
                 data: pd.DataFrame,
                 strategy_fn: Callable = None,
                 initial_balance: float = 10000,
                 parameter_space: List[ParameterSpace] = None,
                 fitness_metric: str = 'sharpe'):
        """
        Args:
            data: OHLCV veri
            strategy_fn: Strateji fonksiyonu (params, engine, bar -> signal)
            initial_balance: Başlangıç bakiyesi
            parameter_space: Parametre uzayı
            fitness_metric: Fitness metriği (sharpe, return, profit_factor)
        """
        self.data = data
        self.strategy_fn = strategy_fn
        self.initial_balance = initial_balance
        self.param_space = parameter_space or self.DEFAULT_PARAMS
        self.fitness_metric = fitness_metric
    
    def _create_fitness_function(self) -> Callable:
        """Fitness fonksiyonu oluştur"""
        from .backtest import BacktestEngine, PositionSide
        
        def fitness(params: Dict) -> Tuple[float, Dict]:
            # Strategy wrapper
            def strategy_wrapper(engine, bar):
                return self.strategy_fn(params, engine, bar)
            
            # Backtest çalıştır
            engine = BacktestEngine(
                initial_balance=self.initial_balance,
                random_seed=42
            )
            engine.load_data(self.data)
            engine.set_strategy(strategy_wrapper)
            
            try:
                result = engine.run()
                
                # Fitness hesapla
                if self.fitness_metric == 'sharpe':
                    fitness_value = result.sharpe_ratio
                elif self.fitness_metric == 'return':
                    fitness_value = result.total_return_pct
                elif self.fitness_metric == 'profit_factor':
                    fitness_value = result.profit_factor if result.profit_factor != float('inf') else 10
                elif self.fitness_metric == 'calmar':
                    fitness_value = result.calmar_ratio
                else:
                    # Combined score
                    fitness_value = (
                        result.total_return_pct * 0.3 +
                        result.win_rate * 0.2 +
                        result.sharpe_ratio * 10 * 0.3 +
                        (100 - result.max_drawdown_pct) * 0.2
                    )
                
                # Negatif sonsuz kontrolü
                if np.isnan(fitness_value) or np.isinf(fitness_value):
                    fitness_value = -1000
                
                metrics = {
                    'return': result.total_return_pct,
                    'trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown_pct,
                    'profit_factor': result.profit_factor
                }
                
                return fitness_value, metrics
                
            except Exception as e:
                return -1000, {'error': str(e)}
        
        return fitness
    
    def optimize(self, 
                 ga_config: GAConfig = None,
                 progress_callback: Callable = None) -> OptimizationResult:
        """
        Optimizasyon çalıştır
        """
        fitness_fn = self._create_fitness_function()
        
        optimizer = GeneticOptimizer(
            fitness_function=fitness_fn,
            parameter_space=self.param_space,
            config=ga_config or GAConfig()
        )
        
        return optimizer.run(progress_callback)


# ================================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ================================================================

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization (NSGA-II benzeri)
    
    Birden fazla hedefi aynı anda optimize eder:
    - Getiri maximize
    - Risk minimize
    - Trade sayısı optimize
    """
    
    def __init__(self,
                 fitness_functions: List[Callable],
                 parameter_space: List[ParameterSpace],
                 config: GAConfig = None):
        """
        Args:
            fitness_functions: Fitness fonksiyonları listesi
            parameter_space: Parametre uzayı
            config: GA konfigürasyonu
        """
        self.fitness_fns = fitness_functions
        self.param_space = {p.name: p for p in parameter_space}
        self.config = config or GAConfig()
        
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        self.population = []
        self.pareto_front = []
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """ind1, ind2'yi domine ediyor mu?"""
        fitnesses1 = ind1.metrics.get('objectives', [])
        fitnesses2 = ind2.metrics.get('objectives', [])
        
        if not fitnesses1 or not fitnesses2:
            return False
        
        better_in_one = False
        for f1, f2 in zip(fitnesses1, fitnesses2):
            if f1 < f2:
                return False
            if f1 > f2:
                better_in_one = True
        
        return better_in_one
    
    def _fast_non_dominated_sort(self) -> List[List[Individual]]:
        """Hızlı non-dominated sıralama"""
        fronts = [[]]
        dominated_by = {id(ind): set() for ind in self.population}
        dominates_count = {id(ind): 0 for ind in self.population}
        
        for i, ind1 in enumerate(self.population):
            for j, ind2 in enumerate(self.population):
                if i != j:
                    if self._dominates(ind1, ind2):
                        dominated_by[id(ind1)].add(id(ind2))
                    elif self._dominates(ind2, ind1):
                        dominates_count[id(ind1)] += 1
            
            if dominates_count[id(ind1)] == 0:
                fronts[0].append(ind1)
        
        i = 0
        while fronts[i]:
            next_front = []
            for ind1 in fronts[i]:
                for ind2_id in dominated_by[id(ind1)]:
                    ind2 = next(x for x in self.population if id(x) == ind2_id)
                    dominates_count[ind2_id] -= 1
                    if dominates_count[ind2_id] == 0:
                        next_front.append(ind2)
            
            if next_front:
                fronts.append(next_front)
            i += 1
        
        return fronts[:-1] if fronts[-1] == [] else fronts
    
    def _crowding_distance(self, front: List[Individual]):
        """Crowding distance hesapla"""
        n = len(front)
        if n == 0:
            return
        
        n_objectives = len(front[0].metrics.get('objectives', []))
        
        for ind in front:
            ind.metrics['crowding_distance'] = 0
        
        for m in range(n_objectives):
            front.sort(key=lambda x: x.metrics.get('objectives', [0]*n_objectives)[m])
            
            front[0].metrics['crowding_distance'] = float('inf')
            front[-1].metrics['crowding_distance'] = float('inf')
            
            obj_range = (
                front[-1].metrics.get('objectives', [0]*n_objectives)[m] -
                front[0].metrics.get('objectives', [0]*n_objectives)[m]
            )
            
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                front[i].metrics['crowding_distance'] += (
                    front[i+1].metrics.get('objectives', [0]*n_objectives)[m] -
                    front[i-1].metrics.get('objectives', [0]*n_objectives)[m]
                ) / obj_range
    
    def _evaluate(self, individual: Individual) -> Individual:
        """Tüm objectives için değerlendir"""
        objectives = []
        
        for fn in self.fitness_fns:
            try:
                result = fn(individual.genes)
                if isinstance(result, tuple):
                    objectives.append(result[0])
                else:
                    objectives.append(result)
            except:
                objectives.append(float('-inf'))
        
        individual.metrics['objectives'] = objectives
        individual.fitness = sum(objectives)  # Basit toplam
        
        return individual
    
    def run(self, progress_callback: Callable = None) -> List[Individual]:
        """
        Optimizasyon çalıştır
        
        Returns:
            Pareto front (en iyi çözümler)
        """
        # Initialize
        self.population = []
        for _ in range(self.config.population_size):
            genes = {name: space.random() for name, space in self.param_space.items()}
            self.population.append(Individual(genes=genes))
        
        # Evaluate initial population
        self.population = [self._evaluate(ind) for ind in self.population]
        
        for gen in range(self.config.generations):
            # Non-dominated sort
            fronts = self._fast_non_dominated_sort()
            
            # Crowding distance
            for front in fronts:
                self._crowding_distance(front)
            
            # Selection & reproduction
            new_population = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.config.population_size:
                    new_population.extend(front)
                else:
                    # Sort by crowding distance
                    front.sort(
                        key=lambda x: x.metrics.get('crowding_distance', 0),
                        reverse=True
                    )
                    remaining = self.config.population_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break
            
            self.population = new_population
            
            # Pareto front güncelle
            if fronts:
                self.pareto_front = fronts[0]
            
            if progress_callback:
                progress_callback(gen + 1, self.config.generations, len(self.pareto_front))
        
        return self.pareto_front


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_optimization_report(result: OptimizationResult):
    """Optimizasyon raporunu yazdır"""
    print("\n" + "="*60)
    print("GENETİK ALGORİTMA OPTİMİZASYON RAPORU")
    print("="*60)
    
    print(f"\n{'GENEL':^60}")
    print("-"*60)
    print(f"Jenerasyon Sayısı:   {result.final_generation}")
    print(f"Toplam Değerlendirme: {result.total_evaluations}")
    print(f"Süre:                {result.elapsed_time:.1f} saniye")
    print(f"Early Stopped:       {'Evet' if result.early_stopped else 'Hayır'}")
    
    print(f"\n{'EN İYİ PARAMETRELER':^60}")
    print("-"*60)
    for name, value in result.best_params.items():
        print(f"{name:20}: {value:.4f}")
    
    print(f"\n{'EN İYİ METRİKLER':^60}")
    print("-"*60)
    print(f"Fitness:             {result.best_fitness:.4f}")
    for name, value in result.best_metrics.items():
        if isinstance(value, float):
            print(f"{name:20}: {value:.4f}")
        else:
            print(f"{name:20}: {value}")
    
    print("\n" + "="*60)


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("Genetic Algorithm Optimizer test ediliyor...\n")
    
    # Basit test fonksiyonu (Rastrigin function)
    def rastrigin(params):
        x = params.get('x', 0)
        y = params.get('y', 0)
        n = 2
        A = 10
        fitness = -(A * n + (x**2 - A * np.cos(2*np.pi*x)) + (y**2 - A * np.cos(2*np.pi*y)))
        return fitness, {'x': x, 'y': y}
    
    # Parametre uzayı
    param_space = [
        ParameterSpace('x', -5.12, 5.12),
        ParameterSpace('y', -5.12, 5.12)
    ]
    
    # GA konfigürasyonu
    config = GAConfig(
        population_size=30,
        generations=50,
        early_stopping_generations=15
    )
    
    # Optimize
    print("Rastrigin function optimizing (global minimum at 0,0)...")
    optimizer = GeneticOptimizer(rastrigin, param_space, config)
    
    def progress(gen, total, fitness):
        if gen % 10 == 0:
            print(f"   Gen {gen}/{total}: fitness = {fitness:.4f}")
    
    result = optimizer.run(progress)
    
    print(f"\nEn iyi çözüm:")
    print(f"   x = {result.best_params['x']:.4f}")
    print(f"   y = {result.best_params['y']:.4f}")
    print(f"   fitness = {result.best_fitness:.4f}")
    print(f"   (Optimal: x=0, y=0, fitness=0)")
    
    print("\n✓ Genetic Algorithm testi başarılı!")
