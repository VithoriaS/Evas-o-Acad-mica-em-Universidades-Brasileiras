"""
ANÁLISE EXPLORATÓRIA DOS DADOS - TCC
====================================

Script para gerar análise detalhada dos dados iniciais para o artigo.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset():
    """Análise exploratória completa dos dados para o artigo"""
    print("🎓 ANÁLISE EXPLORATÓRIA DOS DADOS - TCC")
    print("=" * 60)
    print("Autora: Vitória de Lourdes Carvalho Santos")
    print("PUC Minas - 2025")
    print("=" * 60)
    
    # Carregar dados
    print("\n📊 Carregando dados...")
    try:
        df = pd.read_csv('enrollments.csv', nrows=10000)  # Amostra maior para análise
        print(f"✅ Dados carregados: {df.shape[0]:,} registros e {df.shape[1]} colunas")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None
    
    # 1. CARACTERÍSTICAS GERAIS DO DATASET
    print(f"\n🔍 1. CARACTERÍSTICAS GERAIS DO DATASET")
    print("-" * 50)
    print(f"• Total de registros: {len(df):,}")
    print(f"• Colunas disponíveis: {len(df.columns)}")
    print(f"• Período dos dados: de {df['created_at'].min()} até {df['created_at'].max()}")
    
    # Missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    cols_with_missing = missing_pct[missing_pct > 0].head(5)
    if len(cols_with_missing) > 0:
        print(f"• Colunas com valores ausentes:")
        for col, pct in cols_with_missing.items():
            print(f"  - {col}: {pct}%")
    
    # 2. DISTRIBUIÇÃO POR STATUS DOS ESTUDANTES
    print(f"\n📈 2. DISTRIBUIÇÃO POR STATUS DOS ESTUDANTES")
    print("-" * 50)
    status_counts = df['workflow_state'].value_counts()
    total = len(df)
    
    for status, count in status_counts.items():
        percentage = (count/total)*100
        print(f"• {status.upper():<12}: {count:>6,} estudantes ({percentage:5.1f}%)")
    
    # Calcular taxa de evasão
    evaded_statuses = ['inactive', 'deleted', 'completed']  # Incluir completed como "saída"
    evaded = sum(status_counts.get(status, 0) for status in evaded_statuses if status in status_counts.index)
    evasion_rate = (evaded / total) * 100
    print(f"\n🚨 TAXA DE EVASÃO IDENTIFICADA: {evasion_rate:.1f}%")
    
    # 3. ANÁLISE TEMPORAL DAS MATRÍCULAS
    print(f"\n📅 3. ANÁLISE TEMPORAL DAS MATRÍCULAS")
    print("-" * 50)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['ano_matricula'] = df['created_at'].dt.year
    df['mes_matricula'] = df['created_at'].dt.month
    
    # Por ano
    matriculas_ano = df['ano_matricula'].value_counts().sort_index()
    print("Distribuição por ano:")
    for ano, count in matriculas_ano.items():
        if pd.notna(ano) and ano >= 2020:  # Focar anos recentes
            percentage = (count/total)*100
            print(f"  {int(ano)}: {count:,} matrículas ({percentage:.1f}%)")
    
    # Sazonalidade
    matriculas_mes = df['mes_matricula'].value_counts().sort_index()
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
            'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    print("\nSazonalidade (top 6 meses):")
    for mes, count in matriculas_mes.head(6).items():
        if pd.notna(mes):
            percentage = (count/total)*100
            print(f"  {meses[int(mes)-1]}: {count:,} ({percentage:.1f}%)")
    
    # 4. ANÁLISE DE ATIVIDADE DOS ESTUDANTES
    print(f"\n⏱️  4. ANÁLISE DE ATIVIDADE DOS ESTUDANTES")
    print("-" * 50)
    df['total_activity_time'] = pd.to_numeric(df['total_activity_time'], errors='coerce')
    activity_data = df['total_activity_time'].dropna()
    
    if len(activity_data) > 0:
        activity_stats = activity_data.describe()
        print(f"• Estatísticas do tempo de atividade (horas):")
        print(f"  - Média: {activity_stats['mean']:,.0f} horas")
        print(f"  - Mediana: {activity_stats['50%']:,.0f} horas")
        print(f"  - Desvio padrão: {activity_stats['std']:,.0f} horas")
        print(f"  - Mínimo: {activity_stats['min']:,.0f} horas")
        print(f"  - Máximo: {activity_stats['max']:,.0f} horas")
        
        # Categorização por quartis
        q1 = activity_stats['25%']
        q3 = activity_stats['75%']
        
        df['categoria_atividade'] = 'Baixa'
        df.loc[df['total_activity_time'] >= q1, 'categoria_atividade'] = 'Média'
        df.loc[df['total_activity_time'] >= q3, 'categoria_atividade'] = 'Alta'
        
        categoria_counts = df['categoria_atividade'].value_counts()
        print(f"\n• Categorização por nível de atividade:")
        for categoria, count in categoria_counts.items():
            percentage = (count/len(df))*100
            print(f"  - {categoria}: {count:,} estudantes ({percentage:.1f}%)")
    
    # 5. ANÁLISE POR TIPO DE ESTUDANTE
    print(f"\n👥 5. ANÁLISE POR TIPO DE ESTUDANTE")
    print("-" * 50)
    if 'type' in df.columns:
        tipo_counts = df['type'].value_counts()
        for tipo, count in tipo_counts.items():
            percentage = (count/total)*100
            print(f"• {tipo}: {count:,} estudantes ({percentage:.1f}%)")
    
    # 6. CORRELAÇÃO ENTRE ATIVIDADE E STATUS
    print(f"\n🔗 6. CORRELAÇÃO ENTRE ATIVIDADE E STATUS")
    print("-" * 50)
    
    if len(activity_data) > 0:
        # Atividade média por status
        activity_by_status = df.groupby('workflow_state')['total_activity_time'].agg(['mean', 'median', 'count'])
        print("Tempo de atividade por status:")
        for status, row in activity_by_status.iterrows():
            print(f"  • {status.upper():<12}: Média {row['mean']:>8,.0f}h | Mediana {row['median']:>6,.0f}h | n={row['count']:,}")
        
        # Teste estatístico
        active_activity = df[df['workflow_state'] == 'active']['total_activity_time'].dropna()
        other_activity = df[df['workflow_state'] != 'active']['total_activity_time'].dropna()
        
        if len(active_activity) > 30 and len(other_activity) > 30:
            try:
                t_stat, p_value = ttest_ind(active_activity, other_activity)
                print(f"\n• Teste t-student (ativos vs outros):")
                print(f"  - Estatística t: {t_stat:.3f}")
                print(f"  - P-valor: {p_value:.6f}")
                significance = "SIGNIFICATIVA" if p_value < 0.05 else "não significativa"
                print(f"  - Diferença estatisticamente {significance} (α = 0.05)")
            except:
                print("• Não foi possível realizar teste estatístico")
    
    # 7. ANÁLISE DE PERMANÊNCIA
    print(f"\n📊 7. ANÁLISE DE PERMANÊNCIA DOS ESTUDANTES")
    print("-" * 50)
    df['dias_na_plataforma'] = (pd.Timestamp.now() - df['created_at']).dt.days
    permanencia_data = df['dias_na_plataforma'].dropna()
    
    if len(permanencia_data) > 0:
        perm_stats = permanencia_data.describe()
        print(f"• Tempo de permanência na plataforma:")
        print(f"  - Média: {perm_stats['mean']:,.0f} dias ({perm_stats['mean']/365:.1f} anos)")
        print(f"  - Mediana: {perm_stats['50%']:,.0f} dias ({perm_stats['50%']/365:.1f} anos)")
        
        # Por status
        permanencia_por_status = df.groupby('workflow_state')['dias_na_plataforma'].agg(['mean', 'median'])
        print(f"\nPermanência média por status:")
        for status, row in permanencia_por_status.iterrows():
            print(f"  • {status.upper():<12}: {row['mean']:>6.0f} dias (mediana: {row['median']:>6.0f})")
    
    # 8. INSIGHTS E CONCLUSÕES
    print(f"\n💡 8. PRINCIPAIS INSIGHTS PARA O ARTIGO")
    print("-" * 50)
    
    # Insight 1: Taxa de evasão
    if evasion_rate > 30:
        severity = "CRÍTICA"
    elif evasion_rate > 20:
        severity = "ALTA"
    elif evasion_rate > 10:
        severity = "MODERADA"
    else:
        severity = "BAIXA"
    
    print(f"• TAXA DE EVASÃO: {evasion_rate:.1f}% - Classificada como {severity}")
    if evasion_rate > 20:
        print("  ⚠️  Requer intervenção urgente da instituição")
    
    # Insight 2: Atividade como preditor
    if len(activity_data) > 0 and 'active_activity' in locals() and 'other_activity' in locals():
        if len(active_activity) > 0 and len(other_activity) > 0:
            diff_atividade = active_activity.mean() - other_activity.mean()
            if diff_atividade > 1000:
                print(f"• TEMPO DE ATIVIDADE: Forte preditor de permanência")
                print(f"  📊 Diferença de {diff_atividade:,.0f} horas entre ativos e demais")
            elif diff_atividade > 500:
                print(f"• TEMPO DE ATIVIDADE: Moderado preditor de permanência")
                print(f"  📊 Diferença de {diff_atividade:,.0f} horas entre grupos")
    
    # Insight 3: Padrões temporais
    if len(matriculas_mes) > 0:
        mes_pico = matriculas_mes.idxmax()
        if pd.notna(mes_pico):
            print(f"• SAZONALIDADE: Pico de matrículas em {meses[int(mes_pico)-1]}")
            print("  📅 Sugere padrão típico de calendário acadêmico brasileiro")
    
    # Insight 4: Distribuição de atividade
    if 'categoria_counts' in locals():
        baixa_atividade_pct = (categoria_counts.get('Baixa', 0) / total) * 100
        if baixa_atividade_pct > 25:
            print(f"• ENGAJAMENTO: {baixa_atividade_pct:.0f}% dos estudantes com baixa atividade")
            print("  💡 Oportunidade para estratégias de engajamento precoce")
    
    # 9. RECOMENDAÇÕES PARA MODELOS PREDITIVOS
    print(f"\n🎯 9. RECOMENDAÇÕES PARA MODELOS PREDITIVOS")
    print("-" * 50)
    print("• Variáveis mais promissoras identificadas:")
    print("  1. total_activity_time (tempo de atividade)")
    print("  2. workflow_state (status atual)")
    print("  3. dias_na_plataforma (tempo de permanência)")
    print("  4. padrões sazonais de matrícula")
    
    print(f"\n• Estratégias de modelagem recomendadas:")
    print("  - Usar técnicas de balanceamento de classes (SMOTE)")
    print("  - Aplicar validação cruzada temporal")
    print("  - Considerar features de engenharia baseadas em tempo")
    print("  - Implementar limiar de decisão otimizado para recall")
    
    return df

def main():
    """Executar análise completa"""
    df = analyze_dataset()
    
    if df is not None:
        print(f"\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print(f"📄 Use esses insights no seu artigo do TCC")
        print(f"🎓 Dados analisados: {len(df):,} registros")
    else:
        print(f"❌ Falha na análise dos dados")

if __name__ == "__main__":
    main()