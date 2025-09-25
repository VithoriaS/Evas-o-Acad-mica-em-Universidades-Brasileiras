"""
Predição de Evasão Acadêmica usando Machine    def load_and_merge_data(self, sample_size=1000):

====================================================

Este script implementa modelos preditivos para identificação de estudantes
com risco de evasão acadêmica usando dados do Canvas LMS.

Autores: Vitória de Lourdes Carvalho Santos
Orientador: Wladmir Cardoso Brandao
Instituição: PUC Minas - Instituto de Ciências Exatas e Informática
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Configuração para visualizações
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EvasaoPredictor:
    """
    Classe para predição de evasão acadêmica usando múltiplos algoritmos
    de machine learning.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_and_merge_data(self, sample_size=50000):
        """
        Carrega e integra os dados das diferentes planilhas CSV.
        Para datasets grandes, usa amostragem para evitar problemas de memória.
        
        Args:
            sample_size (int): Tamanho da amostra para datasets grandes
        """
        print("Carregando dados das planilhas...")
        
        try:
            # Carregando as planilhas com otimização de memória
            print("Carregando courses.csv...")
            self.courses = pd.read_csv('courses.csv', low_memory=False)
            
            print("Carregando enrollments.csv...")
            self.enrollments = pd.read_csv('enrollments.csv', low_memory=False)
            
            print("Carregando scores.csv...")
            self.scores = pd.read_csv('scores.csv', low_memory=False)
            
            # Não carregamos assignments por enquanto para economizar memória
            # self.assignments = pd.read_csv('assignments.csv', low_memory=False)
            
            print(f"Courses: {self.courses.shape}")
            print(f"Enrollments: {self.enrollments.shape}")
            print(f"Scores: {self.scores.shape}")
            
            # Verificar se os dados são muito grandes
            total_enrollments = len(self.enrollments)
            if total_enrollments > sample_size:
                print(f"Dataset muito grande ({total_enrollments:,} registros).")
                print(f"Usando amostra aleatória de {sample_size:,} registros...")
                
                # Amostra aleatória estratificada por curso
                self.enrollments = self._sample_data(self.enrollments, sample_size)
                print(f"Amostra de enrollments: {self.enrollments.shape}")
            
        except FileNotFoundError as e:
            print(f"Erro ao carregar arquivos: {e}")
            return None
        except MemoryError as e:
            print(f"Erro de memória ao carregar dados: {e}")
            print("Tente usar um sample_size menor.")
            return None
            
        # Integrando os dados
        self.merged_data = self._merge_tables()
        print(f"Dados integrados: {self.merged_data.shape}")
        
        return self.merged_data
    
    def _optimize_dtypes(self, df):
        """
        Otimiza os tipos de dados para reduzir uso de memória.
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                # Para colunas de texto, converter para category se tiver poucos valores únicos
                unique_count = df[col].nunique()
                if unique_count / len(df) < 0.5:  # Se menos de 50% são únicos
                    df[col] = df[col].astype('category')
        
        new_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memória otimizada: {original_memory:.2f} MB -> {new_memory:.2f} MB ({new_memory/original_memory:.1%})")
        
        return df

    def _sample_data(self, df, sample_size):
        """
        Cria uma amostra estratificada dos dados para reduzir uso de memória.
        """
        if 'course_id' in df.columns and len(df['course_id'].unique()) > 10:
            # Amostra estratificada por curso
            return df.groupby('course_id', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, sample_size // len(df['course_id'].unique()))))
            ).reset_index(drop=True)
        else:
            # Amostra aleatória simples
            return df.sample(min(sample_size, len(df))).reset_index(drop=True)
    
    def _merge_tables(self):
        """
        Realiza a integração das tabelas usando os IDs comuns com otimização de memória.
        """
        print("Realizando merge das tabelas...")
        
        # Otimização: selecionar apenas colunas necessárias antes do merge
        enrollments_cols = ['id', 'user_id', 'course_id', 'workflow_state', 'type', 
                           'created_at', 'updated_at', 'last_activity_at', 'total_activity_time']
        
        scores_cols = ['enrollment_id', 'current_score', 'final_score', 
                      'current_points', 'final_points']
        
        courses_cols = ['id', 'name', 'workflow_state', 'created_at']
        
        # Filtrar colunas existentes
        enrollments_cols = [col for col in enrollments_cols if col in self.enrollments.columns]
        scores_cols = [col for col in scores_cols if col in self.scores.columns]
        courses_cols = [col for col in courses_cols if col in self.courses.columns]
        
        enrollments_subset = self.enrollments[enrollments_cols].copy()
        scores_subset = self.scores[scores_cols].copy()
        courses_subset = self.courses[courses_cols].copy()
        
        # Merge enrollments com scores (inner join para reduzir dados)
        print("Fazendo merge enrollments + scores...")
        data = enrollments_subset.merge(
            scores_subset, 
            left_on='id', 
            right_on='enrollment_id', 
            how='inner',  # Mudado para inner para reduzir dados
            suffixes=('_enroll', '_score')
        )
        
        # Limpar memória
        del enrollments_subset, scores_subset
        
        # Merge com courses
        print("Fazendo merge com courses...")
        data = data.merge(
            courses_subset,
            left_on='course_id',
            right_on='id',
            how='left',
            suffixes=('', '_course')
        )
        
        # Limpar memória
        del courses_subset
        
        return data
    
    def preprocess_data(self):
        """
        Realiza o pré-processamento dos dados com otimização de memória.
        """
        print("Iniciando pré-processamento dos dados...")
        
        if not hasattr(self, 'merged_data'):
            print("Dados não carregados. Execute load_and_merge_data() primeiro.")
            return None
        
        data = self.merged_data.copy()
        
        # Otimização de tipos de dados para economizar memória
        data = self._optimize_dtypes(data)
        
        # Criando variável target (evasão)
        data = self._create_target_variable(data)
        
        # Engenharia de características
        data = self._feature_engineering(data)
        
        # Selecionando características relevantes
        features = self._select_features(data)
        
        # Limpar dados originais da memória
        del data
        
        # Limpeza e tratamento de valores ausentes
        features = self._handle_missing_values(features)
        
        # Codificação de variáveis categóricas
        features = self._encode_categorical_variables(features)
        
        # Separando features e target
        X = features.drop('target_evadido', axis=1)
        y = features['target_evadido']
        
        # Limpar features da memória
        del features
        
        # Normalizando features numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        self.X = X
        self.y = y
        
        print(f"Pré-processamento concluído. Shape final: {X.shape}")
        print(f"Distribuição das classes: {y.value_counts()}")
        
        return X, y
    
    def _create_target_variable(self, data):
        """
        Cria a variável target indicando evasão (1) ou não-evasão (0).
        """
        # Considerando como evadido: workflow_state = 'inactive' ou sem atividade recente
        data['last_activity_recent'] = pd.to_datetime(data['last_activity_at'], errors='coerce')
        current_date = datetime.now()
        
        # Estudante evadiu se:
        # 1. Status inactive/deleted OU
        # 2. Sem atividade nos últimos 6 meses OU  
        # 3. Score final muito baixo (< 20) e sem atividade recente
        data['target_evadido'] = 0
        
        condition1 = data['workflow_state'].isin(['inactive', 'deleted'])
        condition2 = (current_date - data['last_activity_recent']).dt.days > 180
        condition3 = (data['final_score'] < 20) & (data['total_activity_time'] < 500)
        
        data.loc[condition1 | condition2 | condition3, 'target_evadido'] = 1
        
        return data
    
    def _feature_engineering(self, data):
        """
        Cria novas características a partir dos dados existentes.
        """
        print("Criando novas características...")
        
        # Converter datas apenas se existirem
        if 'created_at' in data.columns:
            data['created_at_date'] = pd.to_datetime(data['created_at'], errors='coerce')
            data['days_since_enrollment'] = (datetime.now() - data['created_at_date']).dt.days
            # Remover coluna de data original para economizar memória
            data.drop(['created_at', 'created_at_date'], axis=1, inplace=True)
        else:
            data['days_since_enrollment'] = 30  # Valor padrão
        
        # Taxa de atividade
        if 'total_activity_time' in data.columns:
            data['activity_rate'] = data['total_activity_time'] / (data['days_since_enrollment'] + 1)
        else:
            data['activity_rate'] = 0
        
        # Performance relativa (score vs média do curso)
        if 'current_score' in data.columns and 'course_id' in data.columns:
            course_avg = data.groupby('course_id')['current_score'].mean()
            data['course_avg_score'] = data['course_id'].map(course_avg)
            data['relative_performance'] = data['current_score'] - data['course_avg_score']
            # Limpar coluna temporária
            data.drop(['course_avg_score'], axis=1, inplace=True)
        else:
            data['relative_performance'] = 0
        
        # Variação de desempenho
        if 'current_score' in data.columns and 'final_score' in data.columns:
            data['score_variation'] = ((data['current_score'] - data['final_score']) / 
                                      (data['current_score'] + 1)) * 100
        else:
            data['score_variation'] = 0
        
        # Indicadores de risco
        if 'total_activity_time' in data.columns:
            data['low_engagement'] = (data['total_activity_time'] < 1000).astype(np.int8)
        else:
            data['low_engagement'] = 1
            
        if 'current_score' in data.columns:
            data['poor_performance'] = (data['current_score'] < 50).astype(np.int8)
        else:
            data['poor_performance'] = 1
            
        data['irregular_activity'] = (data['activity_rate'] < 5).astype(np.int8)
        
        return data
    
    def _select_features(self, data):
        """
        Seleciona as características mais relevantes para o modelo.
        """
        selected_features = [
            # Target
            'target_evadido',
            
            # Desempenho acadêmico
            'current_score', 'final_score', 'current_points', 'final_points',
            
            # Atividade e engajamento
            'total_activity_time', 'activity_rate',
            
            # Performance relativa
            'relative_performance', 'score_variation',
            
            # Indicadores de risco
            'low_engagement', 'poor_performance', 'irregular_activity',
            
            # Características do curso
            'workflow_state', 'type'
        ]
        
        # Filtrando apenas colunas que existem
        available_features = [col for col in selected_features if col in data.columns]
        
        return data[available_features]
    
    def _handle_missing_values(self, data):
        """
        Trata valores ausentes no dataset.
        """
        # Para variáveis numéricas: preencher com mediana
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'target_evadido':
                data[col].fillna(data[col].median(), inplace=True)
        
        # Para variáveis categóricas: preencher com moda
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown', 
                           inplace=True)
        
        return data
    
    def _encode_categorical_variables(self, data):
        """
        Codifica variáveis categóricas.
        """
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'target_evadido':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        return data
    
    def train_single_model(self, model_name='Decision Tree'):
        """
        Treina apenas um modelo por vez para economizar memória.
        
        Args:
            model_name: 'Decision Tree', 'Random Forest', ou 'XGBoost'
        """
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            print("Dados não preprocessados. Execute preprocess_data() primeiro.")
            return None
        
        print(f"Iniciando treinamento do modelo: {model_name}")
        
        # Aplicando SMOTE para balanceamento (com sample menor para economizar memória)
        print("Aplicando SMOTE para balanceamento...")
        smote = SMOTE(random_state=42, k_neighbors=3)  # Reduzindo k_neighbors
        X_balanced, y_balanced = smote.fit_resample(self.X, self.y)
        
        print(f"Dados balanceados: {X_balanced.shape}")
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Configurando apenas o modelo solicitado
        model_configs = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8,  # Reduzindo profundidade
                min_samples_split=50,  # Aumentando para simplificar
                min_samples_leaf=20,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduzindo número de árvores
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=1  # Usando apenas 1 core
            ),
            'XGBoost': XGBClassifier(
                learning_rate=0.1,
                max_depth=4,  # Reduzindo profundidade
                n_estimators=100,  # Reduzindo estimadores
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1  # Usando apenas 1 core
            )
        }
        
        if model_name not in model_configs:
            print(f"Modelo '{model_name}' não encontrado. Modelos disponíveis: {list(model_configs.keys())}")
            return None
            
        model = model_configs[model_name]
        self.models = {model_name: model}
        
        print(f"\nTreinando {model_name}...")
        
        # Treinamento
        model.fit(X_train, y_train)
        
        # Predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Validação cruzada simplificada (3 folds para economizar tempo/memória)
        cv_scores = cross_val_score(
            model, X_balanced, y_balanced, 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        self.results = {model_name: metrics}
        
        # Importância das características (se disponível)
        if hasattr(model, 'feature_importances_'):
            feature_names = self.X.columns
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance = {model_name: importance}
        
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\nTreinamento do {model_name} concluído!")
        print(f"Acurácia: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        
        return self.results
    
    def evaluate_single_model(self):
        """
        Avalia o desempenho do modelo treinado.
        """
        if not self.results:
            print("Modelo não treinado. Execute train_single_model() primeiro.")
            return None
        
        print("\n" + "="*50)
        print("RESULTADO DO MODELO PREDITIVO")
        print("="*50)
        
        # Pegar o único modelo treinado
        model_name = list(self.results.keys())[0]
        metrics = self.results[model_name]
        
        print(f"\nModelo: {model_name}")
        print(f"Acurácia: {metrics['accuracy']:.3f}")
        print(f"Precisão: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"Validação Cruzada: {metrics['cv_mean']:.3f} (±{metrics['cv_std']:.3f})")
        
        # Mostrar importância das características se disponível
        if model_name in self.feature_importance:
            print(f"\nTop 5 características mais importantes:")
            top_features = self.feature_importance[model_name].head(5)
            for _, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return metrics
    
    def plot_results(self):
        """
        Cria visualizações dos resultados.
        """
        if not self.results:
            print("Modelos não treinados.")
            return None
        
        # Gráfico de barras com métricas
        metrics_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Acurácia
        axes[0,0].bar(metrics_df.index, metrics_df['accuracy'])
        axes[0,0].set_title('Acurácia dos Modelos')
        axes[0,0].set_ylabel('Acurácia')
        axes[0,0].set_ylim(0, 1)
        
        # F1-Score
        axes[0,1].bar(metrics_df.index, metrics_df['f1_score'])
        axes[0,1].set_title('F1-Score dos Modelos')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_ylim(0, 1)
        
        # AUC-ROC
        axes[1,0].bar(metrics_df.index, metrics_df['auc_roc'])
        axes[1,0].set_title('AUC-ROC dos Modelos')
        axes[1,0].set_ylabel('AUC-ROC')
        axes[1,0].set_ylim(0, 1)
        
        # Importância das características (XGBoost)
        if 'XGBoost' in self.feature_importance:
            importance_data = self.feature_importance['XGBoost'].head(10)
            axes[1,1].barh(importance_data['feature'], importance_data['importance'])
            axes[1,1].set_title('Importância das Características (XGBoost)')
            axes[1,1].set_xlabel('Importância')
        
        plt.tight_layout()
        plt.savefig('resultados_modelos.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_risk_categories(self, model_name='XGBoost'):
        """
        Cria categorias de risco baseadas nas probabilidades do modelo.
        """
        if model_name not in self.models:
            print(f"Modelo {model_name} não encontrado.")
            return None
        
        model = self.models[model_name]
        probabilities = model.predict_proba(self.X_test)[:, 1]
        
        # Definindo categorias de risco
        risk_categories = []
        for prob in probabilities:
            if prob < 0.3:
                risk_categories.append('Baixo Risco')
            elif prob < 0.7:
                risk_categories.append('Médio Risco')
            else:
                risk_categories.append('Alto Risco')
        
        # Contando distribuição
        risk_counts = pd.Series(risk_categories).value_counts()
        risk_percentages = (risk_counts / len(risk_categories) * 100).round(1)
        
        print(f"\nCategorização de Risco ({model_name}):")
        print("-" * 40)
        for category, count in risk_counts.items():
            percentage = risk_percentages[category]
            print(f"{category}: {count} estudantes ({percentage}%)")
        
        return risk_categories, probabilities

def main():
    """
    Função principal para execução do pipeline completo.
    """
    print("Sistema de Predição de Evasão Acadêmica")
    print("="*50)
    
    # Inicializando o preditor
    predictor = EvasaoPredictor()
    
    # Pipeline completo
    try:
        # 1. Carregamento e integração dos dados (usando amostra para datasets grandes)
        data = predictor.load_and_merge_data(sample_size=20000)  # Reduzido para 20k
        if data is None:
            return
        
        print(f"Usando amostra de {len(data):,} registros para análise.")
        
        # 2. Pré-processamento
        X, y = predictor.preprocess_data()
        if X is None:
            return
        
        # 3. Treinamento do modelo (apenas Decision Tree por padrão)
        results = predictor.train_single_model('Decision Tree')
        if results is None:
            return
        
        # 4. Avaliação
        metrics = predictor.evaluate_single_model()
        
        # 5. Categorização de risco
        predictor.create_risk_categories('Decision Tree')
        
        print("\nPipeline executado com sucesso!")
        print("Para treinar outros modelos, use:")
        print("predictor.train_single_model('Random Forest')")
        print("predictor.train_single_model('XGBoost')")
        
    except MemoryError as e:
        print(f"Erro de memória: {e}")
        print("Tente executar novamente com sample_size menor (ex: 10000)")
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()