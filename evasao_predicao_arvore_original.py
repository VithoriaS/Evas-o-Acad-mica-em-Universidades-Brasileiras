"""
==============================================================================
PREDIÇÃO DE EVASÃO ACADÊMICA - ÁRVORE DE DECISÃO COM DADOS ORIGINAIS
==============================================================================
Autor: Vitória de Lourdes Carvalho Santos
Dataset: financeiro.csv (apenas variáveis originais - 36 features)
Objetivo: Modelo Árvore de Decisão baseline para comparação
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

print("🎯 PREDIÇÃO DE EVASÃO - ÁRVORE DE DECISÃO COM DADOS ORIGINAIS")
print("=" * 70)

# ==============================================================================
# FUNÇÕES AUXILIARES (mesmo código do XGBoost original)
# ==============================================================================

def carregar_dados_originais(arquivo='financeiro.csv'):
    """Carrega e prepara os dados originais (sem enriquecimento)"""
    print(f"\n📊 Carregando dados originais do {arquivo}...")
    df = pd.read_csv(arquivo, low_memory=False)
    print(f"Dataset carregado: {len(df):,} registros e {len(df.columns)} colunas")
    return df

def preparar_dados(df, amostra_size=15000):
    """Prepara dados para modelagem"""
    print(f"\n🔧 Preparando dados...")
    
    # Criar variável target (Desistente + Cancelado = Evadido)
    df['ind_evadido'] = df['dsc_situacao_aluno_curso'].str.strip().isin(['Desistente', 'Cancelado']).astype(int)
    
    features_numericas = [
        'qtd_semestres_cursados', 'perc_cursado', 'media_nota_anterior',
        'media_frequencia_anterior', 'qtd_reprovacoes_curso',
        'qtd_disc_reprov_nota_curso', 'qtd_disc_reprov_frequencia_curso',
        'idade_aluno', 'val_distancia_campus'
    ]
    
    features_categoricas = [
        'sexo_aluno', 'dsc_turno', 'dsc_forma_ingresso',
        'ind_possui_bolsa', 'ind_possui_financiamento', 'ind_inadimplente'
    ]
    
    df_clean = df.dropna(subset=features_numericas + features_categoricas + ['ind_evadido'])
    print(f"Registros após limpeza: {len(df_clean):,}")
    print(f"Taxa de evasão: {df_clean['ind_evadido'].mean():.1%}")
    
    if len(df_clean) > amostra_size:
        df_sample = df_clean.groupby('ind_evadido', group_keys=False).apply(
            lambda x: x.sample(n=int(amostra_size * len(x) / len(df_clean)), random_state=42)
        )
        print(f"\nAmostra estratificada: {len(df_sample):,} registros")
    else:
        df_sample = df_clean
    
    X = df_sample[features_numericas + features_categoricas].copy()
    y = df_sample['ind_evadido']
    
    le_dict = {}
    for col in features_categoricas:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    X['performance_combinada'] = X['media_nota_anterior'] * X['media_frequencia_anterior'] / 100
    X['taxa_reprovacao'] = X['qtd_reprovacoes_curso'] / (X['qtd_semestres_cursados'] + 1)
    X['faixa_etaria'] = pd.cut(X['idade_aluno'], bins=[0, 24, 34, 44, 100], labels=[0, 1, 2, 3])
    X['faixa_etaria'] = X['faixa_etaria'].cat.codes
    
    print(f"\n✅ Dataset preparado: {X.shape[0]} amostras, {X.shape[1]} features")
    
    return X, y, features_numericas + features_categoricas + ['performance_combinada', 'taxa_reprovacao', 'faixa_etaria']

def treinar_arvore(X_train, y_train):
    """Treina modelo Árvore de Decisão"""
    print("\n🌳 Treinando modelo Árvore de Decisão...")
    
    # Hiperparâmetros IDÊNTICOS ao modelo enriquecido para comparação justa
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(f"✅ Modelo treinado - Profundidade: {model.get_depth()}, Folhas: {model.get_n_leaves()}")
    
    return model

def avaliar_modelo(model, X_test, y_test, feature_names):
    """Avalia o desempenho do modelo"""
    print("\n📈 Avaliando modelo...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*70)
    print("RESULTADOS - ÁRVORE DE DECISÃO (DADOS ORIGINAIS)")
    print("="*70)
    print(f"Acurácia:  {accuracy:.1%}")
    print(f"Precisão:  {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1-Score:  {f1:.1%}")
    print(f"AUC-ROC:   {auc_roc:.1%}")
    print("="*70)
    
    cm = confusion_matrix(y_test, y_pred)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Features Mais Importantes:")
    print(feature_importance.head(10).to_string(index=False))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }

def gerar_visualizacoes(resultados):
    """Gera gráficos de análise"""
    print("\n📊 Gerando visualizações...")
    
    # Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = resultados['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Não Evadido', 'Evadido'],
                yticklabels=['Não Evadido', 'Evadido'])
    plt.title('Matriz de Confusão - Árvore de Decisão (Dados Originais)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.tight_layout()
    plt.savefig('Matriz_Confusao_Arvore_Original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Matriz de Confusão salva")
    
    # Importância das Features
    plt.figure(figsize=(10, 8))
    top_features = resultados['feature_importance'].head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importância', fontweight='bold')
    plt.title('Top 15 Features - Árvore de Decisão (Dados Originais)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('Importancia_Features_Arvore_Original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Importância das Features salva")
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(resultados['y_test'], resultados['y_pred_proba'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Árvore (AUC = {resultados["auc_roc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classificador Aleatório')
    plt.xlabel('Taxa de Falsos Positivos', fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
    plt.title('Curva ROC - Árvore de Decisão (Dados Originais)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Curva_ROC_Arvore_Original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Curva ROC salva")

def main():
    df = carregar_dados_originais()
    X, y, feature_names = preparar_dados(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Divisão treino/teste:")
    print(f"   Treino: {len(X_train):,} amostras")
    print(f"   Teste:  {len(X_test):,} amostras")
    
    model = treinar_arvore(X_train, y_train)
    resultados = avaliar_modelo(model, X_test, y_test, feature_names)
    gerar_visualizacoes(resultados)
    
    resultados_df = pd.DataFrame({
        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Valor': [
            f"{resultados['accuracy']:.1%}",
            f"{resultados['precision']:.1%}",
            f"{resultados['recall']:.1%}",
            f"{resultados['f1']:.1%}",
            f"{resultados['auc_roc']:.1%}"
        ]
    })
    resultados_df.to_csv('Resultados_Arvore_Original.csv', index=False)
    print("\n✅ Resultados salvos em 'Resultados_Arvore_Original.csv'")
    
    print("\n" + "="*70)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*70)

if __name__ == "__main__":
    main()
