
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)

class EvasaoPredicaoFinal:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.data = None
        
    def load_data(self, sample_size=2000):
        """Carrega dados de enrollments com sample controlado"""
        print(f"📊 Carregando amostra de {sample_size:,} registros...")
        
        try:
            # Carregar dados de enrollments
            self.data = pd.read_csv('enrollments.csv', nrows=sample_size)
            print(f"✅ Dados carregados: {self.data.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            return False
    
    def create_features(self):
        """Cria features para predição"""
        print("🔧 Criando features...")
        
        df = self.data.copy()
        
        # Target: evasão baseada em workflow_state
        df['target_evadido'] = (df['workflow_state'] != 'active').astype(int)
        
        # Features numéricas
        features = []
        
        # Atividade total
        if 'total_activity_time' in df.columns:
            df['total_activity_time'] = pd.to_numeric(df['total_activity_time'], errors='coerce').fillna(0)
            features.append('total_activity_time')
            
            # Feature derivada: baixa atividade
            df['baixa_atividade'] = (df['total_activity_time'] < df['total_activity_time'].median()).astype(int)
            features.append('baixa_atividade')
        
        # Tempo desde criação
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['dias_matriculado'] = (pd.Timestamp.now() - df['created_at']).dt.days
            df['dias_matriculado'] = df['dias_matriculado'].fillna(0)
            features.append('dias_matriculado')
            
            # Feature derivada: matrícula recente
            df['matricula_recente'] = (df['dias_matriculado'] < 365).astype(int)
            features.append('matricula_recente')
        
        # Última atividade
        if 'last_activity_at' in df.columns:
            df['last_activity_at'] = pd.to_datetime(df['last_activity_at'], errors='coerce')
            df['dias_sem_atividade'] = (pd.Timestamp.now() - df['last_activity_at']).dt.days
            df['dias_sem_atividade'] = df['dias_sem_atividade'].fillna(999)
            features.append('dias_sem_atividade')
            
            # Feature derivada: inativo recentemente
            df['inativo_recente'] = (df['dias_sem_atividade'] > 90).astype(int)
            features.append('inativo_recente')
        
        # Codificar variáveis categóricas
        categorical_features = ['type', 'workflow_state']
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                features.append(f'{col}_encoded')
        
        # Preparar dataset final
        final_features = features + ['target_evadido']
        df_final = df[final_features].copy()
        df_final = df_final.dropna()
        
        print(f"✅ Features criadas: {len(features)}")
        print(f"📊 Dataset final: {df_final.shape}")
        print(f"🎯 Distribuição target: {df_final['target_evadido'].value_counts().to_dict()}")
        
        return df_final, features
    
    def train_models(self, df, features):
        """Treina múltiplos modelos"""
        print("\n🤖 Treinando modelos...")
        
        X = df[features]
        y = df['target_evadido']
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # Definir modelos com parâmetros mais restritivos para evitar overfitting
        models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=5,  # Reduzido para evitar overfitting
                min_samples_split=50,  # Aumentado
                min_samples_leaf=20,   # Aumentado
                max_features='sqrt',   # Limitando features
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=30,       # Reduzido
                max_depth=4,          # Mais restritivo
                min_samples_split=100, # Mais restritivo
                min_samples_leaf=25,   # Mais restritivo
                max_features='sqrt',   # Limitando features
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Treinar cada modelo
        for name, model in models.items():
            print(f"\n🔄 Treinando {name}...")
            
            # Treinar
            model.fit(X_train, y_train)
            
            # Predições
            y_pred = model.predict(X_test)
            
            # Métricas
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # Validação cruzada
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"✅ {name} - Acurácia: {metrics['accuracy']:.3f}")
        
        return X_test, y_test
    
    def show_results(self):
        """Exibe resultados detalhados"""
        print("\n" + "="*60)
        print("🏆 RESULTADOS FINAIS DOS MODELOS")
        print("="*60)
        
        # Tabela de resultados
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(3)
        
        print("\n📊 Métricas de Performance:")
        print(results_df[['accuracy', 'precision', 'recall', 'f1_score']])
        
        print("\n🎯 Validação Cruzada:")
        for name, metrics in self.results.items():
            print(f"{name}: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        # Melhor modelo
        best_model = results_df['accuracy'].idxmax()
        best_acc = results_df.loc[best_model, 'accuracy']
        print(f"\n🥇 Melhor modelo: {best_model} (Acurácia: {best_acc:.3f})")
        
        return results_df
    
    def analyze_dropout_profile(self, df):
        """Análise descritiva do perfil dos estudantes evadidos"""
        print("\n" + "="*60)
        print("📊 ANÁLISE DO PERFIL DOS ESTUDANTES EVADIDOS")
        print("="*60)
        
        evadidos = df[df['target_evadido'] == 1]
        nao_evadidos = df[df['target_evadido'] == 0]
        
        print(f"📈 Total de estudantes: {len(df):,}")
        print(f"🚨 Estudantes evadidos: {len(evadidos):,} ({len(evadidos)/len(df)*100:.1f}%)")
        print(f"✅ Estudantes ativos: {len(nao_evadidos):,} ({len(nao_evadidos)/len(df)*100:.1f}%)")
        
        print("\n🔍 COMPARAÇÃO ENTRE GRUPOS:")
        print("-" * 50)
        
        # Análise de atividade
        if 'total_activity_time' in df.columns:
            evadidos_atividade = evadidos['total_activity_time'].mean()
            ativos_atividade = nao_evadidos['total_activity_time'].mean()
            
            print(f"⏰ Tempo médio de atividade:")
            print(f"   • Evadidos: {evadidos_atividade:.0f} horas")
            print(f"   • Ativos: {ativos_atividade:.0f} horas")
            print(f"   • Diferença: {((ativos_atividade - evadidos_atividade) / evadidos_atividade * 100):.1f}% maior para ativos")
        
        # Análise temporal
        if 'dias_matriculado' in df.columns:
            evadidos_tempo = evadidos['dias_matriculado'].mean()
            ativos_tempo = nao_evadidos['dias_matriculado'].mean()
            
            print(f"\n📅 Tempo médio matriculado:")
            print(f"   • Evadidos: {evadidos_tempo:.0f} dias ({evadidos_tempo/365:.1f} anos)")
            print(f"   • Ativos: {ativos_tempo:.0f} dias ({ativos_tempo/365:.1f} anos)")
        
        # Análise de inatividade
        if 'dias_sem_atividade' in df.columns:
            evadidos_inatividade = evadidos['dias_sem_atividade'].mean()
            ativos_inatividade = nao_evadidos['dias_sem_atividade'].mean()
            
            print(f"\n⌛ Dias sem atividade:")
            print(f"   • Evadidos: {evadidos_inatividade:.0f} dias")
            print(f"   • Ativos: {ativos_inatividade:.0f} dias")
        
        # Distribuição por período de inatividade
        print(f"\n� DISTRIBUIÇÃO POR PERÍODO DE INATIVIDADE:")
        print("-" * 40)
        
        inatividade_ranges = [
            (0, 30, "0-30 dias (Ativos recentes)"),
            (31, 90, "31-90 dias (Risco moderado)"), 
            (91, 180, "91-180 dias (Risco alto)"),
            (181, 365, "181-365 dias (Risco crítico)"),
            (366, 999, "Mais de 1 ano (Provável evasão)")
        ]
        
        for min_dias, max_dias, descricao in inatividade_ranges:
            mask = (df['dias_sem_atividade'] >= min_dias) & (df['dias_sem_atividade'] <= max_dias)
            count = df[mask].shape[0]
            evadidos_count = df[mask & (df['target_evadido'] == 1)].shape[0]
            taxa_evasao = (evadidos_count / count * 100) if count > 0 else 0
            
            print(f"   {descricao:<30} {count:>4} estudantes ({taxa_evasao:>5.1f}% evadidos)")
        
        # Padrões de atividade
        print(f"\n🎯 PADRÕES IDENTIFICADOS:")
        print("-" * 30)
        
        # Estudantes com baixa atividade
        baixa_atividade = df['baixa_atividade'].sum()
        baixa_atividade_evadidos = df[df['baixa_atividade'] == 1]['target_evadido'].sum()
        taxa_evasao_baixa = (baixa_atividade_evadidos / baixa_atividade * 100) if baixa_atividade > 0 else 0
        
        print(f"📉 Estudantes com baixa atividade: {baixa_atividade:,}")
        print(f"    • Evadidos neste grupo: {baixa_atividade_evadidos:,} ({taxa_evasao_baixa:.1f}%)")
        
        # Estudantes inativos recentemente
        inativos_recentes = df['inativo_recente'].sum()
        inativos_evadidos = df[df['inativo_recente'] == 1]['target_evadido'].sum()
        taxa_evasao_inativo = (inativos_evadidos / inativos_recentes * 100) if inativos_recentes > 0 else 0
        
        print(f"⏸️  Estudantes inativos (>90 dias): {inativos_recentes:,}")
        print(f"    • Evadidos neste grupo: {inativos_evadidos:,} ({taxa_evasao_inativo:.1f}%)")
        
        return evadidos, nao_evadidos
    
    def show_feature_importance(self):
        """Mostra importância das features"""
        print("\n" + "="*60)
        print("🔍 IMPORTÂNCIA DAS CARACTERÍSTICAS")
        print("="*60)
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                print(f"\n📋 {name}:")
                
                # Pegar feature names do último dataset usado
                df, features = self.create_features()
                
                importance_data = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for _, row in importance_data.head(10).iterrows():
                    print(f"  {row['feature']:<25} {row['importance']:.3f}")
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                print(f"\n📋 {name}:")
                
                # Pegar feature names do último dataset usado
                df, features = self.create_features()
                
                importance_data = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for _, row in importance_data.head(10).iterrows():
                    print(f"  {row['feature']:<25} {row['importance']:.3f}")
    
    def create_visualizations(self):
        """Cria visualizações dos resultados"""
        print("\n📈 Criando visualizações...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Métricas dos modelos
            results_df = pd.DataFrame(self.results).T
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            ax1 = axes[0, 0]
            results_df[metrics].plot(kind='bar', ax=ax1)
            ax1.set_title('Métricas de Performance dos Modelos')
            ax1.set_ylabel('Score')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Distribuição do target
            ax2 = axes[0, 1]
            df, _ = self.create_features()
            target_counts = df['target_evadido'].value_counts()
            ax2.pie(target_counts.values, labels=['Não Evadido', 'Evadido'], autopct='%1.1f%%')
            ax2.set_title('Distribuição das Classes')
            
            # 3. Feature importance (se disponível)
            ax3 = axes[1, 0]
            if 'Random Forest' in self.models:
                model = self.models['Random Forest']
                df, features = self.create_features()
                importance = pd.Series(model.feature_importances_, index=features)
                importance.nlargest(10).plot(kind='barh', ax=ax3)
                ax3.set_title('Importância das Features (Random Forest)')
            
            # 4. Comparison chart
            ax4 = axes[1, 1]
            accuracy_scores = [self.results[model]['accuracy'] for model in self.results.keys()]
            model_names = list(self.results.keys())
            ax4.bar(model_names, accuracy_scores)
            ax4.set_title('Comparação de Acurácia')
            ax4.set_ylabel('Acurácia')
            ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('resultados_evasao.png', dpi=300, bbox_inches='tight')
            print("✅ Gráfico salvo como 'resultados_evasao.png'")
            
        except Exception as e:
            print(f"⚠️ Erro ao criar visualizações: {e}")

def main():
    """Execução principal completa"""
    print("🎓 SISTEMA DE PREDIÇÃO DE EVASÃO ACADÊMICA - TCC")
    print("=" * 60)
    print("Autora: Vitória de Lourdes Carvalho Santos")
    print("Orientador: Wladmir Cardoso Brandao")
    print("PUC Minas - 2025")
    print("=" * 60)
    
    # Inicializar
    predictor = EvasaoPredicaoFinal()
    
    # Pipeline completo
    try:
        # 1. Carregar dados
        if not predictor.load_data(sample_size=5000):  # Aumentar sample para dados mais realistas
            return
        
        # 2. Criar features
        df, features = predictor.create_features()
        if len(df) < 50:
            print("❌ Dados insuficientes para treinamento!")
            return
        
        # 3. Treinar modelos
        X_test, y_test = predictor.train_models(df, features)
        
        # 4. Análise do perfil dos evadidos
        evadidos, nao_evadidos = predictor.analyze_dropout_profile(df)
        
        # 5. Mostrar resultados dos modelos
        results_df = predictor.show_results()
        
        # 6. Importância das features
        predictor.show_feature_importance()
        
        # 7. Visualizações
        predictor.create_visualizations()
        
        # 8. Resumo final
        print("\n" + "="*60)
        print("🎯 RESUMO EXECUTIVO")
        print("="*60)
        
        best_acc = results_df['accuracy'].max()
        total_samples = len(df)
        evaded_samples = df['target_evadido'].sum()
        
        print(f"📊 Total de registros analisados: {total_samples:,}")
        print(f"🚨 Estudantes evadidos identificados: {evaded_samples:,} ({evaded_samples/total_samples*100:.1f}%)")
        print(f"🎯 Melhor acurácia obtida: {best_acc:.1%}")
        print(f"✅ Meta de 80% de acurácia: {'ATINGIDA!' if best_acc >= 0.8 else 'Não atingida'}")
        
        print(f"\n🎉 Execução concluída com sucesso!")
        print(f"📁 Resultados salvos em 'resultados_evasao.png'")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()