#  Run app with 
# $ streamlit run covid_19_overview.py —server.maxUploadSize=1028 # to avoid 

# Maps in this dashboard are created using plotly
# Plotly supports two different kinds of maps:
# Mapbox maps are tile-based maps. If your figure is created with a px.scatter_mapbox, px.line_mapbox, 
# px.choropleth_mapbox or px.density_mapbox function or otherwise contains one or more traces of type 
# go.Scattermapbox, go.Choroplethmapbox or go.Densitymapbox, the layout.mapbox object in your figure 
# contains configuration information for the map itself.
# Geo maps are outline-based maps. If your figure is created with a px.scatter_geo, px.line_geo or px.choropleth 
# function or otherwise contains one or more traces of type go.Scattergeo or go.Choropleth, the layout.geo object 
# in your figure contains configuration information for the map itself.

# How Magic works:

# Any time Streamlit sees either a variable or literal value on its own line, 
# it automatically writes that to your app using st.write()
# Also, magic is smart enough to ignore docstrings. 
# That is, it ignores the strings at the top of files and functions.
# If you prefer to call Streamlit commands more explicitly, you can always turn magic off
# in your ~/.streamlit/config.toml with the following setting:

# [runner]
# magicEnabled = false


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import matplotlib.pyplot as plt
from matplotlib import colors

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go

import calplot
import july

import datatable as dt
from datetime import timedelta
import itertools


from helpers import load_data, load_data_ita, rollup_1day, shift_1day

import locale
#locale.setlocale(locale.LC_TIME, 'it_IT')

# Set dashboard pages configuration
st.set_page_config(
    page_title="COVID-19 Dashboard - Global Data Overview",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",#"collapsed"
    menu_items={
    #'Get Help': 'https://www.extremelycoolapp.com/help',
    #'Report a bug': "https://www.extremelycoolapp.com/bug",
    'About': "# Covid19 dashboard"
    }
    )


# Define style (colors and fonts) for plots background and text:
paper_bgcolor = "white"#"#f63366"
plot_bgcolor = "white"
fontcolor = "black"#"floralwhite"
fontfamily = "Rockwell"

# Set title, subheader
st.title('Pandemia di COVID-19')
st.markdown('La presente applicazione permette: \
    \* la visualizzazione dei dati relativi alla diffusione della \
    malattia in tutto il mondo; \
    \* la visualizzazione dei dati per un continente e per una nazione a scelta; \
    \* la visualizzazione dei dati raccolti dalle regioni e dalle province in Italia; \
    \* il confronto della situazione in diverse nazioni a scelta.')
with st.expander('Clicca (espandi/riduci) - Denominazione della nuova malattia da coronavirus'):
    st.markdown('__Cit. de__ «Portale Italiano delle Classificazioni Sanitarie».')
    st.markdown("Sito web: https://www.reteclassificazioni.it/portal_main.php?&portal_view=home")
    st.markdown("> “ L’11 febbraio 2020, l’Organizzazione Mondiale della Sanità ha deciso di denominare\
     la malattia infettiva causata dal nuovo coronavirus recentemente scoperto COVID-19. \
     COVID è l’acronimo di COronaVirus Disease e non è il nome del coronavirus. Il nome provvisorio \
     del coronoavirus assegnato da OMS è novel-coronavirus 2019, abbreviato in 2019-nCoV.\
     Sempre l’11 Febbraio 2020, il Gruppo di studio sui coronaviurs dell’International Committee on\
     Virus Taxonomy (ICTV) ha proposto su bioRxiv di riconoscere il 2019-nvoV come un _“severe acute\
     respiratory syndrome coronavirus 2 (SARS-CoV-2)”_ sulla base di un’analisi filogenetica di \
     coronavirus correlati ( https://www.biorxiv.org/content/10.1101/2020.02.07.937862v1 ). \
     Secondo un articolo pubblicato su Lancet, ridenominare il coronavirus 2019-nCoV con il nome \
     SARS-CoV-2 crea confusione. Questo nuovo virus era sconosciuto prima dei casi iniziati a Wuhan \
     (Cina) a dicembre 2019 e non se ne conosce l’evoluzione. Gli autori del lavoro propongono di \
     denominarlo HCoV-19 (Human Coronavirus 2019) in modo da mantenere la coerenza con la denominazione \
     della malattia stabilita dall’OMS \
     (ref. https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30419-0/fulltext ). \
     Secondo OMS, l’uso del nome SARS può causare paure eccessive specialmente in ASIA. \
     Per questo motivo, OMS ha iniziato a riferirsi al virus come al _“virus responsabile di COVID-19”_ \
     o _“il virus di COVID-19”_. Nessuna di queste  denominazioni intende sostituire il nome ufficiale \
     del virus deciso da ICTV \
     ( https://www.who.int/emergencies/diseases/novel-coronavirus-2019/technical-guidance/naming-the-coronavirus-disease-(covid-2019)-and-the-virus-that-causes-it ). \
     Gli aggiornamenti sulla malattia COVID-19 e sul nuovo coronavirus si trovano a questo \
     indirizzo: https://www.who.int/emergencies/diseases/novel-coronavirus-2019 .  ” ")

st.markdown('Fonte dei dati mondiali: https://ourworldindata.org/coronavirus')

# Load csv data
# Define parameters to the load_data() function
DATE_COLUMN = 'date'
DATA_URL = ('https://covid.ourworldindata.org/data/owid-covid-data.csv')
columnslist = [
    'excess_mortality', 
    'extreme_poverty',
    'female_smokers',
    'hosp_patients',
    'hosp_patients_per_million',
    'icu_patients',
    'icu_patients_per_million',
    'male_smokers',
    'new_tests',
    'new_tests_per_thousand',
    'new_tests_smoothed',
    'new_tests_smoothed_per_thousand',
    'positive_rate',
    'tests_per_case',
    'total_tests',
    'total_tests_per_thousand',
    'weekly_hosp_admissions',
    'weekly_hosp_admissions_per_million',
    'weekly_icu_admissions',
    'weekly_icu_admissions_per_million'
    ]

with st.spinner('Caricamento dati in corso...'):
    data = load_data(DATA_URL, DATE_COLUMN)
st.success('✅ Dati MONDIALI caricati correttamente')
# Create a text element and let the reader know the data is loading.
#data_load_state = st.text('Loading data...')
#data_load_state = st.text('Caricamento dati in corso...')

# Load all (or some) rows of data into the dataframe.
#data = load_data(DATA_URL, DATE_COLUMN)

# Notify the reader that the data was successfully loaded.
#data_load_state.text("Done! (using st.cache)")

#data_load_state.text("✅ Dati caricati correttamente")

with st.expander("APPROFONDIMENTO TECNICO: elenco delle %s variabili descrittive dei dati mondiali" %(len(data.columns))):
    with st.spinner('Attendere...'):
        st.write({index+1: variable for index, variable in enumerate(data.columns)})
    #st.success('Elenco completo delle variabili presenti nel dataset')


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    Not compatible with non-unique index or columns.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

# Display raw data with checkbox hide/display
#if st.checkbox('Show raw data'):
N = 200
with st.expander('APPROFONDIMENTO TECNICO: anteprima della tabella dei dati mondiali (%s righe mostrate su %s righe totali)'%(N, len(data.index))):
    with st.spinner('Attendere, caricamento anteprima tabella dati mondiali in corso...'):
        props = 'border: 5px solid green'
        float_cols = [col for col in data.head(N).columns if type(data.head(N)[col][0]) is np.float64]
        preview_data_N_rows = data.head(N).reset_index(drop=False)
        st.write(preview_data_N_rows.style.format(formatter='{:,.2f}', subset=float_cols, na_rep='MISSING'
            ).set_properties(
            **{'background-color': 'white', 'color': 'black'}
            ).highlight_null('yellow')
            )

with st.expander('TABELLA - Statistica descrittiva dei dati mondiali'):
    with st.spinner('Attendere, calcolo statistica descrittiva dati mondiali grezzi in corso...'):
        props = 'border: 5px solid green'
        #st.subheader('Statistica descrittiva dei dati grezzi')
        statistic_describe_data = data.describe().T.style.format(
            formatter='{:,.2f}', na_rep='MISSING'
            ).bar(
            subset=["mean"], color="#205ff2"
            ).bar(
            subset=["count"], color="royalblue"
            ).set_properties(
            **{'background-color': 'white',
            'color': 'black'
            }
            ).background_gradient(
            subset=["std"], cmap="Greens"
            ).background_gradient(
            subset=["50%"], cmap="coolwarm"
            ).applymap(
            color_negative_red
            )#.set_table_styles(
            #[{'selector' : '', 'props' : props}] #adding color to border of table
            #)
    st.write(statistic_describe_data)


########## DATI ITALIA REGIONI E PROVINCE
DATA_ITA_REGIONI_URL = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
DATA_ITA_PROVINCE_URL = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"
DATE_COLUMN ='data'


with st.spinner('Caricamento dati Italia REGIONI e PROVINCE in corso...'):
    data_Ita_REGIONI, data_Ita_PROVINCE = load_data_ita(DATA_ITA_REGIONI_URL, DATA_ITA_PROVINCE_URL, DATE_COLUMN)
st.markdown("Fonte dei dati italiani per regione: https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
st.success('✅ Dati ITALIANI REGIONALI caricati correttamente')
st.markdown("Fonte dei dati italiani per provincia: https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv")
st.success('✅ Dati ITALIANI PROVINCIALI caricati correttamente')


# Visualize first and last recorded days dates
data_inizio = data.index[-1].date().strftime(format = '%d/%m/%Y')
data_ultimo_aggiornamento = data.index[0].date().strftime(format = '%d/%m/%Y')
#st.write({'Data ultimo aggiornamento: %s - Data di inizio: %s' %(data_ultimo_aggiornamento, data_inizio)})
#st.write({'Numero di righe: %s - Numero di colonne: %s' %(data.shape[0], data.shape[1])})

# Date Range slider to select a window of start-end dates to preview the data
#cols1, _ = st.columns((1,1))

st.header("Andamento della malattia nel tempo a livello mondiale")
st.subheader("Diffusione del contagio:")
c, c1 = st.columns((1,1))
c.markdown("Data esordio: ")
c.code("%s" %(data_inizio))
c1.markdown("Data ultimo aggiornamento: ")
c1.code("%s" %(data_ultimo_aggiornamento))
st.markdown("Scelta dell'intervallo di tempo da visualizzare:")
slider = st.slider(
    label="Situazione aggiornata per il periodo:",
    min_value=data.index[-1].date(), 
    value=(data.index[-1].date(), data.index[0].date()),
    max_value=data.index[0].date(),
    step=timedelta(days=1),
    key='daterangeslider_0',
    help="Trascinare i pallini per selezionare l'intervallo temporale preferito.",
    format="DD/MM/YYYY")
    #on_change= (callable) An optional callback invoked when this slider's value changes.


data_inizio, data_ultimo_aggiornamento = slider[0], slider[1]
st.code( "Dati riferiti al periodo selezionato: %s - %s" %(data_inizio.strftime('%d %B %Y'), data_ultimo_aggiornamento.strftime('%d %B %Y')))
data_inizio_it, data_ultimo_aggiornamento_it = data_inizio.strftime('%B %Y'), data_ultimo_aggiornamento.strftime('%B %Y')
# Display positive rates around the globe
#st.write("Andamento temporale dei contagi nel mondo, per il periodo %s - %s" %(data_inizio.strftime('%B %Y'), data_ultimo_aggiornamento.strftime('%B %Y')))
df = data.sort_index(ascending=True).copy()
df = df.loc[data_inizio : data_ultimo_aggiornamento]
df['Numero di casi'] = df['total_cases']#.fillna(value=0, inplace=False)

# pl1 = calplot.calplot(
#     data = df[~df['total_cases_per_million'].isna()]['total_cases_per_million'],
#     how = 'mean',
#     cmap = 'Reds',
#     textformat = '{:.0f}',
#     figsize = (16,8),
#     suptitle = "Numero di casi per milione di persone nel mondo (media giornaliera)"
#     )
#st.success('calplot calplot() computed - Numero di casi per milione di persone nel mondo (media giornaliera):')
#pl1[0]
cols = st.columns((1,1))
fig = july.heatmap(
    df[~df['total_cases_per_million'].isna()].index.date,
    data = df[~df['total_cases_per_million'].isna()]['total_cases_per_million'],
    title = 'Numero di casi per milione di persone nel mondo',
    cmap = 'Purples',
    month_grid = True,
    horizontal = True,
    colorbar = True,
    value_label = False,
    date_label = False,
    weekday_label = True,
    month_label = True, 
    year_label = True,
    fontfamily = "monospace",
    fontsize = 6,
    titlesize = 'large',
    dpi = 300
    )
#st.success('july heatmap() computed - Numero di casi per milione di persone nel mondo (media giornaliera):')
with cols[0].expander('CALENDARIO annuale - Numero di casi per milione per milione di persone nel mondo'):
    st.pyplot(fig.figure)

fig1 = july.calendar_plot(
    df[~df['positive_rate'].isna()].index.date,
    data = df[~df['positive_rate'].isna()]['positive_rate'],
    cmap = 'Reds'
    )

with cols[0].expander('CALENDARIO mensile - Tasso di positività nel mondo'):
    #st.success('july calendar_plot() computed - Tasso di positività nel mondo (media giornaliera):')
    st.pyplot(fig1.all().figure)

# Map of the world total cases over time:
 # reverse index order due to a bug in plotly legends and marks
df_mothly_agg = df[['iso_code', 'location', 'Numero di casi']].groupby([pd.Grouper(freq='M'), 'iso_code', 'location']).mean().reset_index().set_index('date')
df_mothly_agg.columns = ['iso_code', 'location', 'Numero di casi (media mensile)']

#st.map(df)
fig = px.choropleth(df_mothly_agg, 
    locations="iso_code", 
    color="Numero di casi (media mensile)", 
    hover_name="location", 
    animation_frame=df_mothly_agg.index.strftime("%B, %Y"),
    color_continuous_scale=px.colors.sequential.Purples,
    animation_group=df_mothly_agg.index.strftime("%B")) # line markers between states)
# fig = px.scatter_geo(
#     df, 
#     locations = "iso_code", 
#     color = "continent",
#     hover_name = "location", 
#     size = "casi totali per milione di persone",
#     animation_frame = df.index.strftime("%b, %Y"),
#     animation_group="location",
#     projection = "natural earth",
#     scope = 'world')
fig.update_geos(
    showcountries = True, 
    countrycolor = "gray",
    )
fig.layout.update(
    transition= {'duration': 0}, 
    width = 700, 
    height = 600, 
    title ='Andamento mensile medio del numero di casi rilevati dal %s, fino al %s.' %(data_inizio_it, data_ultimo_aggiornamento_it))
fig.update_layout(
    margin=dict(t=30, b=0, l=0, r=0),
    paper_bgcolor = paper_bgcolor, 
    plot_bgcolor = plot_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily})

with cols[1].expander('ANIMAZIONE SU MAPPA - Diffusione dei contagi nel mondo'):
    st.plotly_chart(figure_or_data = fig, use_container_width = True)


# Map of the world total cases per million people over time:
df = data.sort_index(ascending=True).copy() # reverse index order due to a bug in plotly legends and marks
df = df.loc[data_inizio : data_ultimo_aggiornamento]
df['casi totali/1M abitanti'] = df['total_cases_per_million']#.fillna(value=0, inplace=False)

df_mothly_agg = df[['iso_code', 'location', 'casi totali/1M abitanti']].groupby([pd.Grouper(freq='M'), 'iso_code', 'location']).mean().reset_index().set_index('date')
df_mothly_agg.columns = ['iso_code', 'location', 'casi totali/1M abitanti (media mensile)']

#st.map(df)
fig = px.choropleth(df_mothly_agg, 
    locations="iso_code", 
    color="casi totali/1M abitanti (media mensile)",
    hover_name="location", 
    animation_frame=df_mothly_agg.index.strftime("%B, %Y"),
    color_continuous_scale=px.colors.sequential.Purples,
    animation_group=df_mothly_agg.index.strftime("%B")) # line markers between states)
# fig = px.scatter_geo(
#     df, 
#     locations = "iso_code", 
#     color = "continent",
#     hover_name = "location", 
#     size = "casi totali per milione di persone",
#     animation_frame = df.index.strftime("%b, %Y"),
#     animation_group="location",
#     projection = "natural earth",
#     scope = 'world')
fig.update_geos(
    showcountries = True, 
    countrycolor = "gray",
    )
fig.layout.update(
    transition= {'duration': 0}, 
    width = 700, 
    height = 600, 
    title ='Andamento mensile medio del num. di casi/milione di abitanti dal %s, al %s' %(data_inizio.strftime('%B %Y'), data_ultimo_aggiornamento.strftime('%B %Y')))
fig.update_layout(
    margin=dict(t=30, b=0, l=0, r=0),
    paper_bgcolor = paper_bgcolor, 
    plot_bgcolor = plot_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily})

with cols[1].expander('ANIMAZIONE SU MAPPA - Prevalenza dei contagi nel mondo'):
    st.plotly_chart(figure_or_data = fig, use_container_width = True)

############################################ ANDAMENTO GLOBALE DELLE VACCINAZIONI - ANIMAZIONI
start = pd.to_datetime(data_ultimo_aggiornamento)
end = pd.to_datetime(data_ultimo_aggiornamento - pd.to_timedelta(1, unit='d'))
end_rng = pd.to_datetime(data_ultimo_aggiornamento - pd.to_timedelta(4, unit='d'))

rng = pd.date_range(start, end_rng, freq='1D')

df = data.copy()

df['latest_people_fully_vaccinated'] = df.groupby(['continent', 'location'])['people_fully_vaccinated'].ffill()
df['latest_people_fully_vaccinated'] = df.groupby(['continent', 'location'])['latest_people_fully_vaccinated'].bfill()
#st.write(df[['continent', 'location', 'latest_people_fully_vaccinated']])
latest_people_fully_vaccinated_on_previous_day = df.groupby(['continent', 'location'])['latest_people_fully_vaccinated'].apply(lambda x: x.shift(periods=1, freq='D', axis='index')).reset_index()
df = df.reset_index().merge(latest_people_fully_vaccinated_on_previous_day, how='left', left_on=['continent', 'location', 'date'], right_on=['continent', 'location', 'date']).set_index('date')

df.eval('latest_people_fully_vaccinated_on_previous_day = latest_people_fully_vaccinated_y', inplace=True)
df.eval('latest_people_fully_vaccinated = latest_people_fully_vaccinated_x', inplace=True)
df.drop('latest_people_fully_vaccinated_y', axis=1, inplace=True)
df.drop('latest_people_fully_vaccinated_x', axis=1, inplace=True)
#st.write(df[['location', 'latest_people_fully_vaccinated', 'latest_people_fully_vaccinated_on_previous_day']])

dataframe = df.sort_values(by='location')

sum_population_start = dataframe.loc[start].population.sum()
sum_population_end = dataframe.loc[end].population.sum()

latest_people_fully_vaccinated = dataframe.loc[start].latest_people_fully_vaccinated.sum()
latest_people_fully_vaccinated_on_previous_day = dataframe.loc[start].latest_people_fully_vaccinated_on_previous_day.sum()


st.subheader('Vaccinazioni complete:')
st.markdown('Per vaccinazione completa si intende la somministrazione di \
    una doppia dose di vaccino per il nuovo coronavirus.')
cols = st.columns((.7,1))

#cols[0].write('Popolazione mondiale %s: %s'%(start.date(), sum_population_start))
#cols[1].write('Popolazione mondiale %s: %s'%(end.date(), sum_population_end))

cols[0].code('Dati cumulativi al %s' %(start.date().strftime('%d %B %Y')))

with cols[1].expander('TABELLA - Vaccinazioni complete nel mondo al %s e al giorno precedente e relativo incremento' %(start.date().strftime('%d %B %Y'))):
    st.write(pd.DataFrame(
        dataframe.loc[start][['latest_people_fully_vaccinated', 'latest_people_fully_vaccinated_on_previous_day']].sum(
        axis=0
        ).rename(
        'Mondo').rename(
        {'latest_people_fully_vaccinated': 'Vaccinazioni complete',
        'latest_people_fully_vaccinated_on_previous_day': 'Vaccinazioni complete al giorno precedente'},
        axis=0)
        ).style.background_gradient(subset=["Mondo"], cmap='OrRd'
        ).format(formatter='{:,.0f}')
        )
    st.write('Incremento: +', pd.DataFrame(dataframe.loc[start][['latest_people_fully_vaccinated', 'latest_people_fully_vaccinated_on_previous_day']].sum(
        axis=0
        )).T.eval('latest_people_fully_vaccinated - latest_people_fully_vaccinated_on_previous_day').iloc[0].astype(int).astype(str))

with st.expander('TABELLA - Vaccinazioni complete di ogni Stato nel mondo al %s e al giorno precedente' %(start.date().strftime('%d %B %Y'))):
    st.write(dataframe.loc[start].sort_values(
        by='location'
        )[['location', 'latest_people_fully_vaccinated', 'latest_people_fully_vaccinated_on_previous_day']].reset_index(
        drop=True
        ).set_index('location').rename(
        {'latest_people_fully_vaccinated': 'Vaccinazioni complete',
        'latest_people_fully_vaccinated_on_previous_day': 'Vaccinazioni complete al giorno precedente'},
        axis=1).T.style.background_gradient( 
        cmap='Greens'
        ).format(formatter='{:,.0f}'
        ).highlight_null('YELLOW')
        )



dataframe.eval(
    """
    newly_fully_vaccinated_people = latest_people_fully_vaccinated - latest_people_fully_vaccinated_on_previous_day
    latest_fully_vaccinated_per_hundred = 100 * latest_people_fully_vaccinated / ((@sum_population_end + @sum_population_start) / 2)
    """,
    inplace=True,
    )

newly_fully_vaccinated_people = dataframe.loc[start].newly_fully_vaccinated_people.sum()

vaccination_goal_per_hundred = 70

# Prepare Gauge chart
Gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = dataframe.loc[start]['latest_fully_vaccinated_per_hundred'].sum(),
    mode = "gauge+number+delta",
    #title = {'text': "%s Vaccination Progress" %(location), 'font': {'size': 24}},
    #title = {'text': ('%s, %s - %% pop. con vaccinazione completa' %(location, df.index[0].date())), 'font': {'size': 12}},
    delta = {'reference': vaccination_goal_per_hundred, 'increasing': {'color': "rebeccapurple"}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': "black"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "darkblue",
        'steps' : [
            #{'range': [0, 60], 'color': "#170143"},
            {'range': [0, 60], 'color': "orangered"},
            {'range': [50, 60], 'color': "orange"},
            {'range': [60, 70], 'color': "yellow"},
            {'range': [70, 80], 'color': "greenyellow"},
            {'range': [80, 100], 'color': "cyan"}],
        'threshold' : {
            'line': {'color': "red", 'width': 4}, 
            'thickness': 0.75, 
            'value': 70}}))
# Set layout
mean_population = ((sum_population_end + sum_population_start) / 2)
Gauge.update_layout(
    title=('Percentuale della popolazione mondiale (~%s persone) con vaccinazione completa - %s' %(f'{mean_population:,.0f}', data_ultimo_aggiornamento.strftime('%d/%m/%Y'))),
    paper_bgcolor = paper_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily})

with st.expander('SOMMARIO GRAFICO TACHIMETRO - Vaccinazioni complete nel mondo al %s' %(start.date().strftime('%d/%m/%Y'))):
    st.plotly_chart(figure_or_data=Gauge, use_container_width=True)

    cols = st.columns((1,1))
    cols[0].metric(
        label="Numero di persone con vaccinazione completa e incremento rispetto al giorno precedente:", 
        value=f'{latest_people_fully_vaccinated:,.0f}', 
        delta=f'{newly_fully_vaccinated_people:,.0f}'
        )

    dataframe = dataframe.loc[start]
    missing_locations = dataframe[(dataframe['latest_people_fully_vaccinated'].isna()) & (dataframe['latest_people_fully_vaccinated_on_previous_day'].isna())]['location']
    if not missing_locations.isna().all():
        cols[1].write('Dati mancanti per i seguenti Paesi:')
        for i in missing_locations:
            cols[1].code(i)




###################################### SIDEBAR

# Define Italian names of continents to be displayed in the sidebar singleselect radio menu
continent_options = {
    'Europe': "Europa",
    'South America': "Sud America",
    'North America': "Nord America",
    'Oceania': 'Oceania',
    'Africa': 'Africa',
    'Asia': 'Asia'
    }

default_continent = 'Europe' # app starts-up with 'Europe' continent selected

AllContinents = data[~data['continent'].isna()].continent.unique()

sidebarfilters = st.sidebar

sidebarfilters.header("Filtri")
# Sidebar filters
continent_to_filter = sidebarfilters.radio(
    #"Which continent would you like to inspect?",
    label=("Continente:"),#Scegliere un continente:
    options=(AllContinents),
    format_func = (lambda x: continent_options.get(x, default_continent)),
    help= "I dati relativi alla malattia nel continente selezionato \
    vengono visualizzati, nel dettaglio: \
    l\' andamento temporale dall\'inizio della pandemia \
    dei valori del tasso di positività e \
    delle vaccinazioni complete \
    e le cinque nazioni col maggior numero di decessi alla data scelta."
)

if not continent_to_filter:
    #st.write('You selected %s' %(continent_to_filter))
    #st.write('Hai selezionato: %s' %(continent_to_filter))
#else:
    #st.write("You didn't select a continent.")
    st.warning("⛔️ Non è stato selezionato alcun continente. Selezionare un continente dal menù a lato ⬅️.")
    st.stop()
elif continent_to_filter:
    st.success('Continente selezionato: %s' %(continent_to_filter))

Continent_filter = data.query('continent == @continent_to_filter')#[data['continent'] == continent_to_filter]

All_locations = list(Continent_filter[~Continent_filter['location'].isna()].location.unique())

# Define italian names for continents to use in the sidebar multiselect menu:
if continent_to_filter == 'Europe':
    continente = 'Europa'
elif continent_to_filter == 'Africa':
    continente = continent_to_filter
elif continent_to_filter == 'Asia':
    continente = continent_to_filter
elif continent_to_filter == 'North America':
    continente = 'Nord America'
elif continent_to_filter == 'South America':
    continente = 'Sud America'
elif continent_to_filter == 'Oceania':
    continente = continent_to_filter

st.header("Andamento della malattia nel tempo in %s" %(continente))


# Sample line charts of positive_rate per continent, in time
source = Continent_filter.loc[data_ultimo_aggiornamento:data_inizio][['positive_rate', 'location']].query("~positive_rate.isna()").sort_index(ascending=True)
#source = Continent_filter[['positive_rate', 'location']][~Continent_filter['positive_rate'].isna()].sort_index(ascending=True)
#source = Continent_filter[~Continent_filter['positive_rate'].isna()]
source['dates'] = source.index.strftime('%Y/%m/%d')
source['years'] = source.index.strftime('%Y')
source['months'] = source.index.strftime('%m')
source['yearsmonths'] = source.index.strftime('%Y/%m')

st.subheader('Diffusione del contagio:')
st.code("Dati riferiti al periodo selezionato: %s - %s" %(data_inizio.strftime('%d %B %Y'), data_ultimo_aggiornamento.strftime('%d %B %Y')))

# Line chart with median and iqr of monthly positive rates
base = alt.Chart(source).encode(
    x=alt.X('yearsmonths:O', title='Anno/mese'),
    color=alt.Color('location:N', title='Stato'))

line = base.mark_line().encode(
    y=alt.Y('median(positive_rate):Q', title='tasso di positività (mediana, iqr/mese)'))

band = base.mark_errorband(extent='iqr').encode(
    y=alt.Y('positive_rate:Q', title='tasso di positività (mediana, iqr/mese)'))

with st.expander('GRAFICO a linee - Tasso di positività, andamento mediano (+ iqr) mensile in %s' %(continente)):
    st.altair_chart((band + line), use_container_width=True)

# Altair Ranged Dot Plot of positive_rate per location in continent between years 2020 and 2021
#st.write(source.groupby(by=['location', 'years', 'months']).positive_rate.mean())
chart = alt.layer(
   data=source
).transform_joinaggregate(mean_positive_rate='mean(positive_rate)',
   groupby=['location', 'years']
)

chart += alt.Chart().mark_line(color='#db646f').encode(
   x='mean_positive_rate:Q',
   y='location:N',
   detail='location:N'
)
# Add points for life expectancy in 1955 & 2000
chart += alt.Chart().mark_point(
   size=100,
   opacity=1,
   filled=True
).encode(
   x=alt.X('mean_positive_rate:Q', title='tasso di positività (media/anno)'),
   y=alt.Y('location:N', title='Stato'),
   tooltip = [
   alt.Tooltip('mean_positive_rate:Q'),
   alt.Tooltip('location:N'),
   alt.Tooltip('years:O')
   ],
   color=alt.Color('years:O',
    title='Anno',
    scale=alt.Scale(
        domain=['2020', '2021'],
        range=['#e6959c', '#911a24']
        )
    )
).interactive()

with st.expander('GRAFICO - Tasso di positività, andamento medio annuale in %s' %(continente)):
    st.altair_chart(chart, use_container_width=True)


# Dati per gli ultimi due giorni da user input:

df = Continent_filter.copy()

# set today default value as the last date of recorded data for continent
today = df.index[0]
st.markdown('Scelta della data da visualizzare:')
c, _ = st.columns((1,2))
# Data selection component for user input today:
today = c.date_input(
        "Situazione aggiornata al: ",
        help="Cliccare sulla data per visualizzare il calendario e selezionare una data preferita.",
        value = df.index[0],
        min_value = df.index[-1],
        max_value = df.index[0])

today = pd.to_datetime(today)

st.code("Dati riferiti al %s" %(today.date().strftime('%d/%m/%Y')))

#st.write('Nuovi casi, nuovi casi per milione di persone, casi totali per milione di persone, decessi totali, decessi totali per milione di persone')
with st.expander('TABELLA - %s - N. di nuovi casi, n. casi totali e n. decessi (normalizzati e non) per ogni Stato in %s' %(today.date().strftime('%d/%m/%Y'), continente)):
    st.write(data.loc[today].query('continent == @continent_to_filter').reset_index(drop=True)[[
        'location', 
        'new_cases', 
        'new_cases_per_million',
        'total_cases_per_million',
        'total_deaths',
        'total_deaths_per_million'
        ]].rename(
            {'location': 'Stato',
            'new_cases': 'Nuovi casi',
            'new_cases_per_million': 'Nuovi casi/1M',
            'total_cases_per_million': 'Casi totali/1M',
            'total_deaths': 'Decessi',
            'total_deaths_per_million': 'Decessi/1M'},
        axis=1
        ).style.format(formatter='{:,.0f}', subset=['Nuovi casi', 'Nuovi casi/1M', 'Casi totali/1M', 'Decessi', 'Decessi/1M']
        ).bar(
        subset=['Nuovi casi/1M'], color="orange").bar(
        subset=["Casi totali/1M"], color="royalblue").bar(
        subset=["Decessi/1M"], color="crimson"
        ).highlight_null('YELLOW')
        )
##########################
st.subheader('Vaccinazioni complete:')

cols = st.columns((1,1))
cols[0].code("Dati cumulativi al %s" %(today.date().strftime('%d/%m/%Y')))
# Gauge per continent
dfcon = data.query('continent == @continent_to_filter')
dfcon['latest_people_fully_vaccinated'] = dfcon.groupby('location')['people_fully_vaccinated'].bfill()
dfcon['latest_people_fully_vaccinated'] = dfcon.groupby('location')['latest_people_fully_vaccinated'].ffill()
latest_people_fully_vaccinated_on_previous_day = dfcon.groupby('location')['latest_people_fully_vaccinated'].apply(lambda x: x.shift(periods=1, freq='D', axis='index')).reset_index()#.latest_people_fully_vaccinated
dfcon = dfcon.reset_index().merge(latest_people_fully_vaccinated_on_previous_day, how='left', left_on=['location', 'date'], right_on=['location', 'date']).set_index('date')
dfcon.eval('latest_people_fully_vaccinated_on_previous_day = latest_people_fully_vaccinated_y', inplace=True)
dfcon.eval('latest_people_fully_vaccinated = latest_people_fully_vaccinated_x', inplace=True)
dfcon.drop('latest_people_fully_vaccinated_y', axis=1, inplace=True)
dfcon.drop('latest_people_fully_vaccinated_x', axis=1, inplace=True)

start = today
end = pd.to_datetime(today -  pd.to_timedelta(1, unit='d'))
rng = pd.date_range(start, end, freq='1D')

sum_population = dfcon.loc[start].population.sum()
latest_people_fully_vaccinated = dfcon.loc[start].latest_people_fully_vaccinated.sum()
latest_people_fully_vaccinated_on_previous_day = dfcon.loc[end].latest_people_fully_vaccinated_on_previous_day.sum()
#dfcon = dfcon.loc[rng]
dfcon.eval(
    """
    newly_fully_vaccinated_people = latest_people_fully_vaccinated - latest_people_fully_vaccinated_on_previous_day
    latest_fully_vaccinated_per_hundred = 100 * latest_people_fully_vaccinated / @sum_population
    """,
    inplace=True,
    )
newly_fully_vaccinated_people = dfcon.loc[start].newly_fully_vaccinated_people.sum()

vaccination_goal_per_hundred = 70

# Prepare Gauge chart
Gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = dfcon.loc[start]['latest_fully_vaccinated_per_hundred'].sum(),
    mode = "gauge+number+delta",
    #title = {'text': "%s Vaccination Progress" %(location), 'font': {'size': 24}},
    #title = {'text': ('%s, %s - %% pop. con vaccinazione completa' %(location, df.index[0].date())), 'font': {'size': 12}},
    delta = {'reference': vaccination_goal_per_hundred, 'increasing': {'color': "rebeccapurple"}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': "black"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "darkblue",
        'steps' : [
            #{'range': [0, 60], 'color': "#170143"},
            {'range': [0, 60], 'color': "orangered"},
            {'range': [50, 60], 'color': "orange"},
            {'range': [60, 70], 'color': "yellow"},
            {'range': [70, 80], 'color': "greenyellow"},
            {'range': [80, 100], 'color': "cyan"}],
        'threshold' : {
            'line': {'color': "red", 'width': 4}, 
            'thickness': 0.75, 
            'value': 70}}))
# Set layout
Gauge.update_layout(
    title=('Percentuale della popolazione con vaccinazione completa in %s - %s' %(continente, start.date().strftime('%d/%m/%Y'))),
    paper_bgcolor = paper_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily})

with st.expander('SOMMARIO GRAFICO TACHIMETRO - Vaccinazioni complete in %s al %s'%(continente, start.date().strftime('%d/%m/%Y'))):
    st.plotly_chart(figure_or_data=Gauge, use_container_width=True)

    cols = st.columns((1,1))
    cols[0].metric(label='Numero di persone con vaccinazione completa', 
        value=f'{latest_people_fully_vaccinated:,.0f}', 
        delta=f'{newly_fully_vaccinated_people:,.0f}')

    dataframe = dfcon.loc[start]
    missing_locations = dataframe[(dataframe['latest_people_fully_vaccinated'].isna()) & (dataframe['latest_people_fully_vaccinated_on_previous_day'].isna())]['location']
    if not missing_locations.isna().all():
        cols[1].write('Dati mancanti per i seguenti Paesi:')
        for i in missing_locations:
            cols[1].code(i)


#st.write('Data selezionata:', today.date())
#st.write({'last day date': today})
#st.subheader('(Top 5 new_deaths locations in: %s)' %(continent_to_filter))

@st.cache()
def filter_n_days(data, ndays):
    '''
    Relies on global variable today set here few lines above, cannot put into the helpers.py file!!
    filter dataset retaining only the last n days (today variable and the day before) 
    (if a country hasn't updated data for these last n days, there will be no data
    for that country in the returned dataset)
    '''
    # import global variable today (last date of recorded data for continent)
    global today
    filtered_df = data[ : today - pd.offsets.Day(ndays)]
    return filtered_df

df_last_two_days = filter_n_days(Continent_filter[today:], 1)

# Cycle through locations to record new deaths based on column total_deaths and shifted total_deaths_on_previous_day:
new_deaths = []
locations = []
total_deaths = []
for location in All_locations:
    # Jersey warning
    # Jersey è l'isola più grande del Canale della Manica, tra il Regno Unito e la Francia. 
    # È una dipendenza autoregolamentata del Regno Unito che fonde cultura inglese e francese.
    # Jersey data has not been updated lately
    if location in np.unique(df_last_two_days['location'].values):
        df_loc = df_last_two_days.query("location == @location")#[df_last_two_days['location'] == location]
        #if df_loc.index[0] == today:
        shifted_df_loc = shift_1day(df_loc, 'total_deaths')
        df_loc["new_deaths"] = shifted_df_loc.eval("total_deaths - total_deaths_on_previous_day", inplace=False)#shifted_df_loc["total_deaths"] - shifted_df_loc["total_deaths_on_previous_day"]
        df_loc["total_deaths_on_previous_day"] = shifted_df_loc["total_deaths_on_previous_day"]
        #st.write(df_loc[["location", "new_deaths", "total_deaths_on_previous_day", "total_deaths"]])
        # append data only if there's a record for today with respect to yesterday
        if not df_loc["total_deaths_on_previous_day"].isna().all():
            locations.append(location)
            new_deaths.append(df_loc["new_deaths"][0])
            total_deaths.append(df_loc['total_deaths'][0])
        
# Construct a dict from locations and new deaths lists
new_deaths_dict_Continent_last_day = dict(zip(locations, new_deaths))
# Construct a dict from locations and total deaths lists
total_deaths_dict_Continent_last_day = dict(zip(locations, total_deaths))


from collections import Counter
# import Counter to find out top most values in dict
c = Counter(new_deaths_dict_Continent_last_day)
top_5_new_deaths = dict(c.most_common(5))
# create dict with top 5 new_deaths locations total deaths
top_5_new_deaths_total_deaths = {key: value for (key, value) in total_deaths_dict_Continent_last_day.items() if key in top_5_new_deaths.keys()}


# Display 5 columns widget metrics (KPI style) in dashboard
st.subheader('Stati in %s col maggior numero di decessi registrati il %s:' %(continente, today.strftime('%d/%m/%Y')))
st.caption("Per ogni Stato è indicato:\n- in nero, il n. totale dei decessi dovuti alla COVID-19 fino al %s, \
dall'inizio della pandemia;\n - in verde, il n. totale dei decessi registrati il %s." %(today.strftime('%d/%m/%Y'), today.strftime('%d/%m/%Y'))) # il giorno precedente: (today-timedelta(days=1)).date())

metricscolumns = st.columns(len(top_5_new_deaths))
for index, location in enumerate(top_5_new_deaths.keys()):
    if ( pd.isna(top_5_new_deaths_total_deaths[location]) or pd.isna(top_5_new_deaths[location]) ): # string formatting for nan as decimal integer (%u) without decimals raises error
        metricscolumns[index].subheader('%s°'%(index+1))
        metricscolumns[index].metric(
            label = '%s' %(location), 
            value = ('%s'%(top_5_new_deaths_total_deaths[location])), 
            delta = ('%s'%(top_5_new_deaths[location])))
    else:
        metricscolumns[index].subheader('%s°'%(index+1))
        metricscolumns[index].metric(
            label = '%s' %(location), 
            value = (f'{top_5_new_deaths_total_deaths[location]:,.0f}'), 
            delta = ('%u'%(top_5_new_deaths[location])))
#st.caption('(Total deaths per country with increment of new deaths in last day:)')


Europe_filter = data[data['continent'] == 'Europe']

Italy_filter = data[data['location'] == 'Italy']

Russia_filter = data[data['location'] == 'Russia']


st.header('Italia')

fig = july.heatmap(
    Italy_filter[~Italy_filter['total_cases_per_million'].isna()].index.date,
    data = Italy_filter[~Italy_filter['total_cases_per_million'].isna()]['total_cases_per_million'],
    title = 'Numero di casi per milione di persone in Italia',
    cmap = 'BuPu',
    colorbar = True,
    value_label = False,
    fontsize = 6,
    month_grid = True,
    horizontal = True,
    dpi = 300
    )
#st.success('july heatmap() computed - Tasso di positività in Italia')
st.pyplot(fig.figure)


fig1 = july.calendar_plot(
    Italy_filter[~Italy_filter['positive_rate'].isna()].index,
    data = Italy_filter[~Italy_filter['positive_rate'].isna()]['positive_rate'],
    cmap = 'Reds',
    dpi = 300
    )
with st.expander('CALENDARIO mensile - Tasso di positività in Italia'):
    st.success('july calendar_plot() computed - Tasso di positività in Italia')
    st.pyplot(fig1.all().figure)

fig2 = july.heatmap(
    Italy_filter[~Italy_filter['people_fully_vaccinated'].isna()].index,
    data = Italy_filter[~Italy_filter['people_fully_vaccinated'].isna()]['people_fully_vaccinated'],
    title = 'Vaccinazioni complete in Italia',
    cmap = 'Greens',
    colorbar = True,
    value_label = False,
    date_label = False,
    weekday_label = True,
    month_label = True, 
    year_label = True,
    fontfamily = "monospace",
    fontsize = 6,
    titlesize = 'large',
    dpi = 1080
    )
#st.success('july heatmap() computed - Numero di persone con vaccinazione completa in Italia')
st.pyplot(fig2.figure)


########################## DATI ITALIA REGIONI e PROVINCE #########################
st.header('Dati delle regioni e delle province d\'Italia')

with st.expander('APPROFONDIMENTO TECNICO: elenco delle %s variabili descrittive dei dati italiani regionali' %(len(data_Ita_REGIONI.columns))):
    st.write({index+1: variable for index, variable in enumerate(data_Ita_REGIONI.columns)})

with st.expander('APPROFONDIMENTO TECNICO: elenco delle %s variabili descrittive dei dati italiani provinciali' %(len(data_Ita_PROVINCE.columns))):
    st.write({index+1: variable for index, variable in enumerate(data_Ita_PROVINCE.columns)})


N = 200
counter = 0
for ddff in [data_Ita_REGIONI, data_Ita_PROVINCE]:
    if counter == 0:
        nome = 'regionali'
    elif counter == 1:
        nome = 'provinciali'
    with st.expander('APPROFONDIMENTO TECNICO: - Anteprima della tabella dei dati italiani %s (prime %s righe, su %s totali)'%(nome, N, len(ddff.index))):
        with st.spinner('Attendere caricamento anteprima dati grezzi in corso...'):
            props = 'border: 5px solid green'
            float_cols = [col for col in ddff.head(N).columns if (type(ddff.head(N)[col].values[0]) is np.float64)]
            int_cols = [col for col in ddff.head(N).columns if (type(ddff.head(N)[col].values[0]) is np.int32)]
            preview_data_N_rows = ddff.head(N).reset_index(drop=False)
            st.write(preview_data_N_rows.style.format(formatter='{:,.2f}', subset=float_cols
                ).format(formatter='{:,.0f}', subset=int_cols
                ).highlight_null('yellow')
                )
    counter += 1


data_Ita_PROVINCE.sort_index(ascending=True, inplace=True)
dff = data_Ita_PROVINCE.query('~totale_casi.isna()')
fig = px.choropleth(dff,
    locations="codice_provincia",
    geojson='https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_provinces.geojson', 
    featureidkey='properties.prov_istat_code_num',
    color='totale_casi',
    hover_name="denominazione_provincia", 
    animation_frame=dff.index.strftime("%b, %Y"),
    color_continuous_scale=px.colors.sequential.Reds,  
    animation_group=dff.index)
fig.update_geos(fitbounds="locations", visible=False)
fig.layout.update(transition= {'duration': 1})
fig.update_layout(
    paper_bgcolor = paper_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily},
    updatemenus=[dict(type='buttons',
        showactive=False,
        y=-0.1,
        x=0.1,
        xanchor='right',
        yanchor='top')
    ]
    )
with st.expander('ANIMAZIONE SU MAPPA e GRAFICO a barre - totale casi per provincia in Italia'):
    st.plotly_chart(figure_or_data = fig, use_container_width = True)


data_Ita_REGIONI.sort_index(ascending=True, inplace=True)
dff = data_Ita_REGIONI.query('~totale_positivi_test_molecolare.isna()')
fig = px.choropleth(dff,
    locations="codice_regione",
    geojson='https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson', 
    featureidkey='properties.reg_istat_code_num',
    color='totale_positivi_test_molecolare',
    hover_name="denominazione_regione", 
    animation_frame=dff.index.strftime("%b, %Y"),
    color_continuous_scale=px.colors.sequential.Reds,  
    animation_group=dff.index)
fig.update_geos(fitbounds="locations", visible=False)
fig.layout.update(transition= {'duration': 1})
fig.update_layout(
    paper_bgcolor = paper_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily},
    updatemenus=[dict(type='buttons',
        showactive=False,
        y=-0.1,
        x=0.1,
        xanchor='right',
        yanchor='top')
    ]
    )
# position the slider (closer to the plot)
fig['layout']['sliders'][0]['pad'] = dict(r = 20, t = 0.0,)

fig2 = px.scatter(dff, 
    x="long", 
    y="lat",
    size="totale_positivi_test_molecolare", 
    color="totale_positivi_test_molecolare",
    color_continuous_scale=px.colors.sequential.Reds,
    hover_name="denominazione_regione",
    animation_frame=dff.index.strftime('%b, %Y'),
    size_max=60)
fig2.update_layout(
    paper_bgcolor = paper_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily},
    updatemenus=[dict(type='buttons',
        showactive=False,
        y=-0.1,
        x=-0.1,
        xanchor='right',
        yanchor='top'
        )]
    )
# position the slider (closer to the plot)
fig2['layout']['sliders'][0]['pad'] = dict(r = 20, t = 0.0,)

with st.expander('ANIMAZIONE SU MAPPA e GRAFICO a pallini - totale_positivi_test_molecolare in Italia'):
    cols = st.columns((1,1))
    cols[0].plotly_chart(figure_or_data = fig, use_container_width = True)
    cols[1].plotly_chart(figure_or_data = fig2, use_container_width = True)

data_Ita_REGIONI.sort_index(ascending=False, inplace=True)
dff = data_Ita_REGIONI.query('~isolamento_domiciliare.isna()')
fig1 = px.choropleth(dff,
    locations="codice_regione",
    geojson='https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson', 
    featureidkey='properties.reg_istat_code_num',
    color='isolamento_domiciliare',
    hover_name="denominazione_regione", 
    animation_frame=dff.index.strftime("%b, %Y"),
    color_continuous_scale=px.colors.sequential.Plasma,  
    animation_group=dff.index)
fig1.update_geos(fitbounds="locations", visible=False)
fig1.layout.update(transition= {'duration': 1})
fig1.update_layout(
    paper_bgcolor = paper_bgcolor, 
    font = {'color': fontcolor, 'family': fontfamily})

with st.expander('ANIMAZIONE SU MAPPA - isolamento_domicialiare in Italia'):
    st.plotly_chart(figure_or_data = fig1, use_container_width = True)




#########################################################################
# Menù laterale selezione Stato

sidebarfilters.header("Stato")
location_to_filter = sidebarfilters.multiselect(
    #"Which location would you like to inspect? \n If all, type/select 'All locations'",
    label="Stato",
    #(np.append(Continent_filter[~Continent_filter['location'].isna()].location.unique(), 'All locations'))
    options=(np.append(All_locations, ('Tutti gli Stati (%s)'%(continente)))),
    help=("Seleziona tutti con l'opzione: Tutti gli Stati (%s)" %(continente))
)

if not location_to_filter:
    st.warning("⛔️ Non è stato selezionato alcuno Stato. Per ulteriori dettagli, selezionare uno o più Stati nel menù laterale ⬅️.")
    st.stop()
st.success('Stati selezionati: %s' %(location_to_filter))
#if "All locations" in location_to_filter:
#    location_to_filter = list(Continent_filter[~Continent_filter['location'].isna()].location.unique())

if 'Tutti gli Stati (%s)' %(continente) in location_to_filter:
    location_to_filter = list(Continent_filter[~Continent_filter['location'].isna()].location.unique())


Location_filter = Continent_filter[Continent_filter['location'].isin(location_to_filter)]


st.header('Stato %s: panoramica' %(', '.join(location_to_filter)))

#c1, c2, c3 = st.columns((1, 1, 1))

#@st.cache(suppress_st_warning=True)
def plot_all_locations_vaccinations(datadf, continent_to_filter, location_to_filter):
    '''
    Yields Bars, BarSubheader, Gauge, GaugeSubheader, value, delta for data per location in continent, today:
    2 plotly charts, Bars and Gauge, to be plotted with st.plotly_chart(figure_or_data = fig, use_container_width = True)
    * bar chart and header
    * gauge chart and header
    value, delta for a Metrics/KPI streamlit component for each location in location_to_filter:
    1. a bar chart for complete vaccinations over population over time.
    2. header for the bar chart
    3. a gauge chart of % of population with complete vaccinations as of 'today' - global, from user input - date.
    4. header for the gauge chart 
    5. value for metric (showing total number of people fully vaccinated as of 'today' 
    and newly fully vaccinated people from the day before.)
    6. delta for metric (showing total number of people fully vaccinated as of 'today' 
    and newly fully vaccinated people from the day before.)
    This function relies on user input global variable 'today' as well as on sidebar filters for columns continent and 
    location of datadf (data, from load_data()). 
    Mandatory arguments 
    * 'datadf': raw data loaded using the load_data() function.
    * 'continent_to_filter': user input from sidebar filter radio select related to the column continent of datadf.
    * 'location_to_filter': list of locations user input from sidebar filter multiselect dropdown related to \
    the column location of datadf.
    '''
    #c2.subheader("Gaps from Vaccination Goals over time (70% - 100% population)")
    #c2.subheader("Distanza dal raggiungimento dagli obiettivi di Vaccinazione (del 70% o del 100% della popolazione):")
    # Filter selected continent data
    global today # import global user input variable in function
    
    dfcon = datadf.query("continent == @continent_to_filter")#datadf[datadf['continent'] == continent_to_filter]
    # Define goals for vaccination
    vaccination_goal_per_hundred = 70
    hundred = 100
    
    for location in location_to_filter:
        #with st.expander(location):
        #c1, c2, c3 = st.columns((1, 1, 1))
        #c3.subheader('%s '%(location)) #\n- Progresso verso gli obiettivi di vaccinazione (70%% - 100%% pop.)
        #c3.caption('Panoramica vaccinazioni')

        # Prepare data
        df = dfcon.query("location == @location")#dfcon[dfcon['location'] == location]
        df = rollup_1day(df, 'people_fully_vaccinated')
        dtindex_first_date_in_location = df.index[0]
        # Check if the last date recorded for country is actualy the same as the last date in complete database, 
        # otherwise add new rows with NAN values for each missing day to reach today date, 
        # and then upfill() df with the last registered value for column people_fully_vaccinated 
        # (kind of filling missing records!! (with dates, instead of values).)
        if dtindex_first_date_in_location != today:
            dtindex_first_value_occurrence = df.index[0]
            first_value_people_fully_vaccinated = df.iloc[0].loc['people_fully_vaccinated']
            first_value_latest_people_fully_vaccinated = df.iloc[0].loc['latest_people_fully_vaccinated']
            value_iso_code = df.iloc[0].loc['iso_code']
            value_population = df.iloc[0].loc['population']
            # increment by one day
            dtindex_first_value_occurrence = dtindex_first_value_occurrence + timedelta(days=1)
            # iteration exclusive of dtindex_first_value_occurrence (cycle from a day after the first recorded day up-to today)
            while dtindex_first_value_occurrence <= today:
                df.loc[dtindex_first_value_occurrence] = list(np.repeat(np.nan, len(df.columns)))
                df.loc[dtindex_first_value_occurrence, 'people_fully_vaccinated'] = first_value_people_fully_vaccinated
                df.loc[dtindex_first_value_occurrence, 'latest_people_fully_vaccinated'] = first_value_latest_people_fully_vaccinated
                df['location'].fillna(value=location, inplace=True)
                df['continent'].fillna(value=continent_to_filter, inplace=True)
                df['iso_code'].fillna(value=value_iso_code, inplace=True)
                df['population'].fillna(value=value_population, inplace=True)
                dtindex_first_value_occurrence = dtindex_first_value_occurrence + timedelta(days=1)
            # sort the index! new rows are now at the bottom of df
            df = df.sort_index(ascending=False)
            
        df = shift_1day(df, 'latest_people_fully_vaccinated')
        if not df['latest_people_fully_vaccinated'].isnull().all():
            # Define measures for plotting
            df.eval(
                """
                newly_fully_vaccinated_people = latest_people_fully_vaccinated - latest_people_fully_vaccinated_on_previous_day
                latest_fully_vaccinated_per_hundred = 100 * latest_people_fully_vaccinated / population
                """,
                inplace=True,
                )

            #st.write('newly_fully_vaccinated_people: %s'%(df['newly_fully_vaccinated_people']))
            #st.write('latest_fully_vaccinated_per_hundred: %s'%(df['latest_fully_vaccinated_per_hundred']))
            #df["newly_fully_vaccinated_people"] = df["latest_people_fully_vaccinated"] - df["latest_people_fully_vaccinated_on_previous_day"]
            #df["latest_fully_vaccinated_per_hundred"] = (
            #    100 * df["latest_people_fully_vaccinated"] / df["population"]
            #)
            # Define how far we're from goal of 70/100 vaccinated today
            df.eval(
                """
                Gaps_from_goal = @vaccination_goal_per_hundred - latest_fully_vaccinated_per_hundred
                Gaps_from_total = @hundred - latest_fully_vaccinated_per_hundred
                """,
                inplace=True,
                )
            #df["Gaps_from_goal"] = (
            #    vaccination_goal_per_hundred - df["latest_fully_vaccinated_per_hundred"]
            #)
            # Define how far we're from 100/100 vaccinated today
            #df["Gaps_from_total"] = (
            #    hundred - df["latest_fully_vaccinated_per_hundred"]
            #)
            # Prepare bar plot
            #df = df.sort_values(by='Gaps_from_total', ascending=False)
            x = df.index[~df['Gaps_from_total'].isna()]
            y = df.query("~Gaps_from_total.isna()")['Gaps_from_total']
            #y = df['Gaps_from_total'][~df['Gaps_from_total'].isna()]
            y1 = df.Gaps_from_goal[~df.Gaps_from_goal.isna()]
            Bars = go.Figure(data = [
                #go.Bar(name='over_70%', x=x, y=y1, marker=dict(color = '#170143')),
                go.Bar(name = '% pop.', 
                    x = x, 
                    y = np.round(100-y, 2), 
                    marker = dict(
                        color = np.round(100-y, 2),
                        colorscale = 'greens')
                    )
                #go.Bar(name='over_100%', x=x, y=-y, marker=dict(color = 'cyan'))]
                #go.Bar(name='100% della pop.', x=x, y=100-y, marker=dict(color = 'royalblue'))
                ]
                )
            Bars.update_yaxes(
                showgrid = True,
                title = '% pop. con vaccinazione completa', 
                ticklabelposition = "inside bottom", 
                tick0 = 10, 
                dtick = 10,
                tickvals = [50, 60, 70, 80, 100],
                ticksuffix = '%',
                tickangle = None, 
                tickfont = dict(family = fontfamily, color = fontcolor)# size=14)
                )
            Bars.update_xaxes(
                tickangle = 45
                )
            # Change the bar mode
            Bars.update_layout(
                yaxis_range = [0,100],
                #barmode ='group', 
                plot_bgcolor = plot_bgcolor, 
                paper_bgcolor = paper_bgcolor,
                bargap = 0, 
                #title=('% popolazione con vaccinazione completa'),
                font = {'color': fontcolor, 'family': fontfamily},
                legend = {'valign': 'bottom'}
                #title = {'text': '% Population NOT fully vaccinated over time'}
                #title = {'text': '%s, Obiettivi di vaccinazione'%(location)},
                )
            Bars.add_hline(y = 50, line_width = 3, line_dash = "dash", line_color = "orangered")
            Bars.add_hline(y = 60, line_width = 3, line_dash = "dash", line_color = "orange")
            Bars.add_hline(y = 70, line_width = 3, line_dash = "dash", line_color = "yellow")
            Bars.add_hline(y = 80, line_width = 3, line_dash = "dash", line_color = "greenyellow")
            Bars.add_hline(y = 100, line_width = 3, line_dash = "dash", line_color = "cyan")
            #Bars.add_hrect(y0=00, y1=50, line_width=0, fillcolor="orangered", opacity=0.1)
            Bars.add_hrect(y0=50, y1=60, line_width=0, fillcolor="orange", opacity=0.1)
            Bars.add_hrect(y0=60, y1=70, line_width=0, fillcolor="yellow", opacity=0.1)
            Bars.add_hrect(y0=70, y1=80, line_width=0, fillcolor="greenyellow", opacity=0.1)
            Bars.add_hrect(y0=80, y1=100, line_width=0, fillcolor="cyan", opacity=0.1)

            # Bar Plot
            #c2.write({'Latest People Fully Vaccinated': df.iloc[0,:].loc["latest_people_fully_vaccinated"],
            #    'Total Population': df.iloc[0,:].loc["population"]})
            #c2.write({'Numero di persone con vaccinazione completa': df.iloc[0,:].loc["latest_people_fully_vaccinated"],
            #    ('Popolazione totale %s'%(location)): df.iloc[0,:].loc["population"]})
            BarSubheader = str(location)
            #c3.subheader('')
            #c3.write('Popolazione')
            #c3.plotly_chart(figure_or_data = fig, use_container_width = True)
            
            # Prepare metrics of newly_fully_vaccinated_people
            # c1.write("Nuove vaccinazioni al %s" %(today.date()))

            # Prepare Gauge chart
            Gauge = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = df['latest_fully_vaccinated_per_hundred'][today],
                mode = "gauge+number+delta",
                #title = {'text': "%s Vaccination Progress" %(location), 'font': {'size': 24}},
                #title = {'text': ('%s, %s - %% pop. con vaccinazione completa' %(location, df.index[0].date())), 'font': {'size': 12}},
                delta = {'reference': vaccination_goal_per_hundred, 'increasing': {'color': "rebeccapurple"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': "black"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "darkblue",
                    'steps' : [
                        #{'range': [0, 60], 'color': "#170143"},
                        {'range': [0, 60], 'color': "orangered"},
                        {'range': [50, 60], 'color': "orange"},
                        {'range': [60, 70], 'color': "yellow"},
                        {'range': [70, 80], 'color': "greenyellow"},
                        {'range': [80, 100], 'color': "cyan"}],
                    'threshold' : {
                        'line': {'color': "red", 'width': 4}, 
                        'thickness': 0.75, 
                        'value': 70}}))
            # Set layout
            Gauge.update_layout(
                title=('% pop. con vaccinazione completa'),
                paper_bgcolor = paper_bgcolor, 
                font = {'color': fontcolor, 'family': fontfamily})
            
            if today in df.index:
                value = df.loc[today]["latest_people_fully_vaccinated"]
                delta = df.loc[today]["newly_fully_vaccinated_people"]
            else:
                value = str("❌ Nessun dato rilevato in data: %s. \
                                Per visualizzare il numero di vaccinazioni complete, \
                                scegliere un'altra data." %(today.date()))
                delta = str("❌ Nessun dato rilevato in data: %s. \
                    Per visualizzare il numero di vaccinazioni complete, \
                    scegliere un'altra data." %(today.date()))

            GaugeSubheader = str('%s - Situazione del %s' %(location, today.strftime('%d/%b/%Y')))# df.index[0].date())
            yield Bars, BarSubheader, Gauge, GaugeSubheader, value, delta
            #yield fig1
                # # Gauge Chart
                # c2.subheader('%s' %(location))# df.index[0].date())))
                # c2.write('%s' %(df.index[0].date()))

                # c2.plotly_chart(figure_or_data = fig1, use_container_width = True)
                
                # # Metric new deaths per location
                # if location in new_deaths_dict_Continent_last_day:
                #     c2.metric(label = 'Numero di decessi' ,value = new_deaths_dict_Continent_last_day[location])
                # else:
                #     c2.write('Nessun dato registrato relativo al numero dei decessi.')
                
            
            #yield value
            #yield delta
                # if today in df.index:
                #     if ( pd.isna(df.loc[today]["latest_people_fully_vaccinated"]) or pd.isna(df.loc[today]["newly_fully_vaccinated_people"]) ) : # string formatting for nan values as decimal integers without decimals (%u) raises error
                #         c3.metric(label = ('Numero di persone con vaccinazione completa, nuove vaccinazioni complete'), 
                #             value = ('%s'%(df.loc[today]["latest_people_fully_vaccinated"])), 
                #             delta = ('%s'%(df.loc[today]["newly_fully_vaccinated_people"])))
                #     else:
                #         c3.metric(label = ('Numero di persone con vaccinazione completa, nuove vaccinazioni complete'), 
                #             value = ('%u'%(df.loc[today]["latest_people_fully_vaccinated"])), 
                #             delta = ('%u'%(df.loc[today]["newly_fully_vaccinated_people"])))
                # else:
                #     c3.write("❌ Nessun dato rilevato in data: %s. \
                #         Per visualizzare il numero di vaccinazioni complete, \
                #         scegliere un'altra data." %(today.date()))
            # else:
            #     c2.write("⛔️ Attenzione! Nessun dato rilevato relativamente al numero di persone con vaccinazione completa per lo Stato: %s." %(location))

#c2, c3 = st.columns((1, 1))
#c2.empty()
#c3.empty()
geo_variable_to_be_displayed = 'people_fully_vaccinated'
if not continent_to_filter == 'Oceania':
    if location_to_filter:
        df = Location_filter.sort_index(ascending = False).query('~people_fully_vaccinated.isna()')
        if not df[geo_variable_to_be_displayed].isnull().all():
            with st.expander('ANIMAZIONE SU MAPPA - Vaccinazioni complete in %s' %(', '.join(location_to_filter))):
                fig = px.choropleth(df, locations="iso_code", color=geo_variable_to_be_displayed, hover_name="location", animation_frame=df.index.strftime("%d, %b, %Y"),
                 color_continuous_scale=px.colors.sequential.Plasma, scope=continent_to_filter.lower(), animation_group="location")
                # Plot Map showing total_cases_per_million for all location_to_filter locations toghether along time
                # fig = px.scatter_geo(
                #     df, 
                #     locations = "iso_code", 
                #     color = "location",
                #     hover_name = "location", 
                #     size = geo_variable_to_be_displayed,
                #     animation_frame = df.index.strftime("%Y/%m"),
                #     projection = "kavrayskiy7",
                #     scope = continent_to_filter.lower()
                #     )
                fig.layout.update(transition= {'duration': 1})
                fig.update_layout(
                    paper_bgcolor = paper_bgcolor, 
                    font = {'color': fontcolor, 'family': fontfamily})
                st.plotly_chart(figure_or_data = fig, use_container_width = True)
                
                # Then plot each location gauge and bar charts and metrics of fully vaccinated people
                #plot_all_locations_vaccinations(data, continent_to_filter, location_to_filter)
            
        else:
            st.markdown("❌ Nessun dato riportato relativamente alla variabile _%s_, da %s - %s." %(geo_variable_to_be_displayed, continente, ', '.join(location_to_filter)))
            #plot_all_locations_vaccinations(data, continent_to_filter, location_to_filter)
    else:
        #st.write("You didn't select a location.")
        st.write("⛔️ Attenzione! Non è stato selezionato alcuno Stato. Selezionare uno o più Stati dal menù a lato ⬅️.")


else:
    if location_to_filter:
        df = Location_filter.sort_index(ascending = True).query('~people_fully_vaccinated.isna()')
        if not df[geo_variable_to_be_displayed].isnull().all():
            st.subheader('Vaccinazioni complete')
            st.success('%s' %(geo_variable_to_be_displayed))
            # Plot Map showing total_cases_per_million for all location_to_filter locations toghether along time
            fig = px.choropleth(df, locations="iso_code", color=geo_variable_to_be_displayed, hover_name="location", animation_frame=df.index.strftime("%d, %b, %Y"),
             color_continuous_scale=px.colors.sequential.Plasma, scope='world', animation_group=df.index.strftime("%B"))
            # fig = px.scatter_geo(
            #     df, 
            #     locations = "iso_code", 
            #     color = "location",
            #     hover_name = "location", 
            #     size = "positive_rate",
            #     animation_frame = df.index.strftime("%Y/%m"),
            #     projection = "kavrayskiy7",
            #     scope = 'world'
            #     )
            fig.layout.update(transition= {'duration': 1})
            fig.update_layout(
                paper_bgcolor = paper_bgcolor, 
                font = {'color': fontcolor, 'family': fontfamily})
            st.plotly_chart(figure_or_data = fig, use_container_width = True)
            
            # Then plot each location gauge and bar charts and metrics of fully vaccinated people
            #plot_all_locations_vaccinations(data, continent_to_filter, location_to_filter)
        else:
            st.write("❌ Nessun dato rilevato, relativamente a %s in: %s - %s." %(geo_variable_to_be_displayed, continente, ', '.join(location_to_filter)))
            #plot_all_locations_vaccinations(data, continent_to_filter, location_to_filter)
    else:
        #st.write("You didn't select a location.")
        st.write("⛔️ Attenzione! Non è stato selezionato alcuno Stato. Selezionare uno o più Stati dal menù a lato ⬅️.")

figures = {}
for location in np.sort(location_to_filter):
    df_location = data.query('location == @location')
    if not df_location['positive_rate'].isna().all():
        fig = july.heatmap(
            df_location[~df_location['positive_rate'].isna()].index.date,
            data = df_location[~df_location['positive_rate'].isna()]['positive_rate'],
            title = 'Tasso di positività in %s (media giornaliera)' %(location),
            cmap = 'BuPu',
            colorbar = True,
            value_label = False,
            fontsize = 6,
            month_grid = True,
            horizontal = True,
            dpi = 300
            )
        figures[location] = fig
    elif df_location['positive_rate'].isna().all():
        st.warning('Nessuna informazione rilevata per il tasso di positività in %s.' %(location))

if len(figures) > 0:
    with st.expander('Tasso di positività in %s (media giornaliera)' %(', '.join(figures.keys()))):
        for location, fig in figures.items():
            #st.success('july heatmap() computed - Tasso di positività in %s (media giornaliera)' %(location))
            st.pyplot(fig.figure)

c2, c3 = st.columns((1, 1))

#new_deaths_dict_Continent_last_day
#total_deaths_dict_Continent_last_day

# Plots per location:
for Bars, BarSubheader, Gauge, GaugeSubheader, value, delta in plot_all_locations_vaccinations(data, continent_to_filter, np.sort(location_to_filter)):
    with c2.expander('%s'%(GaugeSubheader)):
        # Gauge Chart
        #st.subheader('%s' %(location))# df.index[0].date())))
        #st.write('%s' %(df.index[0].date()))
        st.plotly_chart(figure_or_data = Gauge, use_container_width = True)
        # Metric new deaths per location
        if BarSubheader in new_deaths_dict_Continent_last_day:
            st.metric(label = "Nuovi decessi nell'ultimo giorno:", value = f'{new_deaths_dict_Continent_last_day[BarSubheader]:,.0f}')
            st.metric(label = 'Totale decessi:', value = f'{total_deaths_dict_Continent_last_day[BarSubheader]:,.0f}')
        else:
            st.write('Nessun dato relativo al numero dei decessi è stato registrato in questa data.')
    with c3.expander('%s - Andamento temporale'%(BarSubheader)):
        #fig = next(figures)
        # Bar chart
        st.plotly_chart(figure_or_data = Bars, use_container_width = True)

        #latest_people_fully_vaccinated_today = next(figures)
        #newly_fully_vaccinated_people_today = next(figures)
        
        if (pd.isna(value) or pd.isna(delta)):
            st.metric(label = ("Numero di persone con vaccinazione completa (e incremento nell'ultimo giorno):"),
                value = (f'{value:,.0f}'), 
                delta = (f'{delta:,.0f}'))
        elif (type(value) == str and type(delta) == str):
            st.write(f'{value:,.0f}')
        elif not (pd.isna(value) and pd.isna(delta)):
            st.metric(label = ("Numero di persone con vaccinazione completa (e incremento nell'ultimo giorno):"), 
                value = (f'{value:,.0f}'), 
                delta = (f'{delta:,.0f}'))



st.header('Confronto fra Stati')
# c, c1 = st.columns((1,1))

# c.code('Andamento del tasso di positività:')
# # Compare selected countries in different continents
# my_chart = c.line_chart(None)

# c1.code('Andamento del numero di nuovi casi rilevati:')
# my_chart_new_cases = c1.line_chart(None)

# dt = data[data['continent'] == 'Africa']
# multilocations['Africa'] = c.multiselect('Continente: Africa', dt[~dt['location'].isna()].location.unique())
# dt = data[data['continent'] == 'Asia']
# multilocations['Asia'] = c.multiselect('Continente: Asia', dt[~dt['location'].isna()].location.unique())
# dt = data[data['continent'] == 'Europe']
# multilocations['Europe'] = c.multiselect('Continente: Europe', dt[~dt['location'].isna()].location.unique())
# dt = data[data['continent'] == 'North America']
# multilocations['North America'] = c.multiselect('Continente: North America', dt[~dt['location'].isna()].location.unique())
# dt = data[data['continent'] == 'South America']
# multilocations['South America'] = c.multiselect('Continente: South America', dt[~dt['location'].isna()].location.unique())
# dt = data[data['continent'] == 'Oceania']
# multilocations['Oceania'] = c.multiselect('Continente: Oceania', dt[~dt['location'].isna()].location.unique())


# Fill  multilocations with user inputs from one menu of States for each continent: 
# - time consuming, page reloads for every new choice
# with st.form(key='Stati da confrontare:'):
#     multilocations = {}
#     for continent in data.query("~continent.isna()")['continent'].unique():#data[~data['continent'].isna()].continent.unique():
#         dt = data.query("continent == @continent")#data[data['continent'] == continent]
#         multilocations[continent] = c.multiselect('%s:' %(continent), dt.query("~location.isna()")['location'].unique())#[~dt['location'].isna()].location.unique())
#     submit_button = st.form_submit_button(label='Aggiorna grafici')

# if submit_button:
#     c1.success('Confronto fra i seguenti Stati: %s'%(str([', '.join(i) for i in multilocations.values() if len(i)>0])))
#     for continent, locations in multilocations.items():
        
#         dt = data.query("continent == @continent")#data[( data['continent'] == continent )]
        
#         if len(locations) > 1:
#             for location in locations:
                
#                 dtt = dt.query("location == @location")['positive_rate']#dt[( dt['location'] == location )].positive_rate
#                 #dt = dt[dt['location'].isin(location)].positive_rate
#                 dtt = dtt.rename(('%s, %s'%(continent, location)))
               
#                 dtnc = dt.query("location == @location")['new_cases']
#                 dtnc = dtt.rename(('%s, %s'%(continent, location)))
                
#                 my_chart.add_rows(dtt)
#                 my_chart_new_cases.add_rows(dtnc)
#         elif len(locations) == 1:

#             dtt = dt.query("location == @locations[0]")['positive_rate']#dt[( dt['location'] == locations[0])].positive_rate
#             #dt = dt[dt['location'] == location].positive_rate
#             dtt = dtt.rename(('%s, %s'%(continent, locations[0])))
            
#             dttnc = dt.query("location == @locations[0]")['new_cases']
#             dttnc = dttnc.rename(('%s, %s'%(continent, locations[0])))
            
#             my_chart.add_rows(dtt)
#             my_chart_new_cases.add_rows(dttnc)


                
# else:
#     st.stop()
#     c1.warning("Scegliere uno o più Stati e premere 'Confronta i dati di: %s'"%(str(multilocations.values())))



def form_callback_confronto_stati():
    ContinentsStates = {}
    AfricaStates = st.session_state.AfricaStates_multiselect
    AsiaStates = st.session_state.AsiaStates_multiselect
    SudAmericaStates = st.session_state.SouthAmericaStates_multiselect
    EuropaStates = st.session_state.EuropeStates_multiselect
    NordAmericaStates = st.session_state.NorthAmericaStates_multiselect
    OceaniaStates = st.session_state.OceaniaStates_multiselect
    ContinentsStates['Africa'] = AfricaStates
    ContinentsStates['Asia'] = AsiaStates
    ContinentsStates['South America'] = SudAmericaStates
    ContinentsStates['Europe'] = EuropaStates
    ContinentsStates['North America'] = NordAmericaStates
    ContinentsStates['Oceania'] = OceaniaStates
    
    container = st.container()
    c, c1 = container.columns((1,1))
    c.code('Tasso di positività:')
    my_chart_pos_rate = c.line_chart(None)
    c1.code('Nuovi casi per milione di abitanti:')
    my_chart_new_cases = c1.line_chart(None)
    c.code('Nuovi decessi per milione di abitanti:')
    my_chart_new_deaths = c.line_chart(None)
    c1.code('Vaccinazioni complete (percentuale):')
    my_chart_people_fully_vac_per_h = c1.line_chart(None)
    
    pos_rs = {}
    new_cs = {}
    new_ds = {}
    tot_vacs = {}
    for i, (continent, locations) in enumerate(ContinentsStates.items()):
        if len(locations) > 0:
            dt = data.query("continent == @continent")
            for location in locations:
                df = dt.query("location == @location")
                pos_r = df['positive_rate']
                new_c = df['new_cases_per_million']
                new_d = df['new_deaths_per_million']
                tot_vac = df['people_fully_vaccinated_per_hundred']
                pos_rs[location] = pos_r
                new_cs[location] = new_c
                new_ds[location] = new_d
                tot_vacs[location] = tot_vac
    my_chart_pos_rate.add_rows(pos_rs)
    my_chart_new_cases.add_rows(new_cs)
    my_chart_new_deaths.add_rows(new_ds)
    my_chart_people_fully_vac_per_h.add_rows(tot_vacs)
    return my_chart_pos_rate, my_chart_new_cases, my_chart_new_deaths, my_chart_people_fully_vac_per_h
    
with st.form(key='Confronto Stati a scelta'):
    cols = st.columns(6)
    continents = np.sort(data.query("~continent.isna()")['continent'].unique())
    ContinentsStates = {}
    for i, continent in enumerate(continents):
        key_continent = continent.replace(" ", "")
        dt = data.query("continent == @continent")
        ContinentsStates[continent] = cols[i].multiselect(
            label = '%s:' %(continent),
            options = dt.query("~location.isna()")['location'].unique(),
            key=('%sStates_multiselect'%(key_continent))
            )
    submit_button = st.form_submit_button(
        label='Avvia Confronto'
        #on_click=form_callback_confronto_stati
        )


if submit_button:
    st.success('Confronto gli Stati: %s' %(str([', '.join(i) for i in ContinentsStates.values() if len(i) > 0])))
    with st.spinner('Attendere...'):
        my_chart, my_chart_new_cases, my_chart_new_deaths, my_chart_people_fully_vac_per_h = form_callback_confronto_stati()


#else:
#    st.stop()
#    st.warning("Scegliere gli Stati da confrontare.")

# Maps per continent over time:
# Uncomment to visualize a single animated map with cases over time per each continent
# for continent in data[~data['continent'].isna()].continent.unique():
#     if continent.lower() in ['africa', 'asia', 'europe', 'north america', 'south america', 'usa', 'world']:
#         st.header(continent)
#         df = data[(~data.total_cases_per_million.isna()) & (data.continent == continent)].sort_index(ascending=False)
#         #st.map(scatter_map, zoom='whole world')# data to be plotted with lat, lng or latitude, longitude columns)
#         fig = px.scatter_geo(
#             df,
#             locations="iso_code",
#             color="location",
#             hover_name="location", 
#             size="total_cases_per_million",
#             animation_frame=df.index.strftime("%Y/%m"),
#             projection="natural earth",
#             scope=continent.lower())
#         fig.layout.update(transition= {'duration': 10})
#         st.plotly_chart(figure_or_data=fig)
#     else:
#         st.header(continent)
#         df = data[(~data.total_cases_per_million.isna()) & (data.continent == continent)].sort_index(ascending=False)
#         #st.map(scatter_map, zoom='whole world')# data to be plotted with lat, lng or latitude, longitude columns)
#         fig = px.scatter_geo(
#             df,
#             locations="iso_code", 
#             color="location",
#             hover_name="location", 
#             size="total_cases_per_million",
#             animation_frame=df.index.strftime("%Y/%m"),
#             projection="natural earth",
#             scope='world')
#         fig.layout.update(transition= {'duration': 10})
#         st.plotly_chart(figure_or_data=fig)

st.header('ML trials')
st.subheader('lazypredict')
from lazypredict.Supervised import (  # pip install lazypredict
    LazyClassifier,
    LazyRegressor,
)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data and split
X, y = load_boston(return_X_y=True)
st.code('X, y = load_boston(return_X_y=True)')
cols = st.columns((1,1))
cols[0].markdown('X set')
cols[0].write(X)
cols[1].markdown('y target')
cols[1].write(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
cols[0].code('Sets: X_train and X_test')
cols[1].code('targets: y_train and y_test')
cols[0].markdown('X training set')
cols[0].write(X_train)
cols[1].markdown('y training target')
cols[1].write(y_train)
cols[0].markdown('X validation set')
cols[0].write(X_test)
cols[1].markdown('y validation target')
cols[1].write(y_test)

# Fit LazyRegressor
reg = LazyRegressor(
    ignore_warnings=True, random_state=1121218, verbose=False
  )
models, predictions = reg.fit(X_train, X_test, y_train, y_test)  # pass all sets

st.code('\
# Fit LazyRegressor\n\
reg = LazyRegressor(\
    ignore_warnings=True, random_state=1121218, verbose=False\
  )\n\
models, predictions = reg.fit(X_train, X_test, y_train, y_test)  # pass all sets')
st.code('models')
st.write(models)
st.code('predictions')
st.write(predictions)

st.markdown('Trying to predict the value of terapia_intensiva in the Italian regions dataset using LazyRegressor')
X = data_Ita_REGIONI.query('terapia_intensiva == terapia_intensiva and ricoverati_con_sintomi == ricoverati_con_sintomi and \
isolamento_domiciliare == isolamento_domiciliare and \
nuovi_positivi == nuovi_positivi and \
dimessi_guariti == dimessi_guariti and \
deceduti == deceduti and \
ingressi_terapia_intensiva == ingressi_terapia_intensiva and \
totale_casi == totale_casi and \
casi_testati == casi_testati and \
codice_regione == codice_regione').loc[:,['ricoverati_con_sintomi',
'isolamento_domiciliare',
'nuovi_positivi',
'dimessi_guariti',
'deceduti',
'ingressi_terapia_intensiva',
'totale_casi',
'casi_testati',
'codice_regione']]
y = data_Ita_REGIONI.query('terapia_intensiva == terapia_intensiva and ricoverati_con_sintomi == ricoverati_con_sintomi and \
isolamento_domiciliare == isolamento_domiciliare and \
nuovi_positivi == nuovi_positivi and \
dimessi_guariti == dimessi_guariti and \
deceduti == deceduti and \
ingressi_terapia_intensiva == ingressi_terapia_intensiva and \
totale_casi == totale_casi and \
casi_testati == casi_testati and \
codice_regione == codice_regione').terapia_intensiva

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
cols = st.columns((1,1))
cols[0].code('Sets: X_train and X_test')
cols[1].code('targets: y_train and y_test')
cols[0].markdown('X training set')
cols[0].write(X_train)
cols[1].markdown('y training target')
cols[1].write(y_train)
cols[0].markdown('X validation set')
cols[0].write(X_test)
cols[1].markdown('y validation target')
cols[1].write(y_test)
# LazyRegressor
models, predictions = reg.fit(X_train, X_test, y_train, y_test)  # pass all sets
st.code('models')
st.write(models)
st.code('predictions')
st.write(predictions)

# Fit LazyClassifier
clas = LazyClassifier(
    ignore_warnings=True, random_state=1121218, verbose=False
  )

st.markdown('Trying to predict the value of codice_regione in the Italian regions dataset using LazyClassifier')
X = data_Ita_REGIONI.query('codice_regione == codice_regione and \
    ricoverati_con_sintomi == ricoverati_con_sintomi and \
    isolamento_domiciliare == isolamento_domiciliare and \
nuovi_positivi == nuovi_positivi and \
dimessi_guariti == dimessi_guariti and \
deceduti == deceduti and \
ingressi_terapia_intensiva == ingressi_terapia_intensiva and \
totale_casi == totale_casi and \
casi_testati == casi_testati').loc[:,['ricoverati_con_sintomi',
'isolamento_domiciliare',
'nuovi_positivi',
'dimessi_guariti',
'deceduti',
'ingressi_terapia_intensiva',
'totale_casi',
'casi_testati']]
y = data_Ita_REGIONI.query('codice_regione == codice_regione and \
    ricoverati_con_sintomi == ricoverati_con_sintomi and \
isolamento_domiciliare == isolamento_domiciliare and \
nuovi_positivi == nuovi_positivi and \
dimessi_guariti == dimessi_guariti and \
deceduti == deceduti and \
ingressi_terapia_intensiva == ingressi_terapia_intensiva and \
totale_casi == totale_casi and \
casi_testati == casi_testati').codice_regione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# LazyClassifier
models, predictions = clas.fit(X_train, X_test, y_train, y_test)  # pass all sets
st.code('models')
st.write(models)
st.code('predictions')
st.write(predictions)



# Optuna is a next-generation automatic hyperparameter tuning framework designed 
# to work on virtually any model and neural network available in today’s ML and Deep learning packages.
# Pros:
# Platform-agnostic: has APIs to work with any framework, including XGBoost, \
# LightGBM, CatBoost, Sklearn, Keras, TensorFlow, PyTorch, etc.
# A large suite of optimization algorithms with early stopping and pruning features baked in
# Easy parallelization with little or no changes to the code
# Built-in support to visually explore tuning history and the importance of each hyperparameter.
# My most favorite feature is its ability to pause/resume/save search histories. 
# Optuna keeps track of all previous rounds of tuning, and you can resume the search for 
# however long you want until you get the performance you want.
# Besides, you can make Optuna RAM-independent for massive datasets and searching 
# by storing results in a local or a remote database by adding an extra parameter.

st.subheader('Optuna')
import optuna  # pip install optuna
st.markdown('Optuna is a next-generation automatic hyperparameter tuning framework designed \
    to work on virtually any model and neural network available in today’s ML and Deep learning packages.\
    ')

st.markdown('For the sake of simplicity, we are trying to optimize the function (x — 1)² + (y + 3)².\
As you can see, the tuned values for x and y are pretty close to the optimal (1, -3).')
def objective(trial):
    x = trial.suggest_float("x", -7, 7)
    y = trial.suggest_float("y", -7, 7)
    return (x - 1) ** 2 + (y + 3) ** 2

st.write('Define function for which optuna is going to optimize the parameters')
st.code('def objective(trial):\n\
    x = trial.suggest_float("x", -7, 7)\n\
    y = trial.suggest_float("y", -7, 7)\n\
    return (x - 1) ** 2 + (y + 3) ** 2')

study = optuna.create_study()
st.code('study = optuna.create_study()')
study.optimize(objective, n_trials=200)  # number of iterations
st.code('study.optimize(objective, n_trials=200) # number of iterations')

st.code('study.best_params')
study.best_params
#{'x': 1.0292346846493052, 'y': -2.969875637298915}

st.code('study.best_value')
study.best_value
#0.0017621440146908432

# SHAP (Shapley Additive exPlanations) is an approach to explain how a model works using 
# concepts from game theory. At its score, SHAP uses something called Shapley values to explain:
# - Which features in the model are the most important
# - The model’s decisions behind any single prediction. 
# For example, asking which features led to this particular output.

# The most notable aspects of SHAP are its unified theme and unique plots that break 
# down the mechanics of any model and neural network. Here is an example plot that shows the 
# feature importances in terms of Shapley values for a single prediction:

st.subheader('SHAP')
st.markdown('SHAP (Shapley Additive exPlanations) is an approach to explain how a model works using\
    concepts from game theory. At its score, SHAP uses something called Shapley values to explain:\n\
    * Which features in the model are the most important\n\
    * The model’s decisions behind any single prediction.\n\
    For example, asking which features led to this particular output.')
import shap  # pip install shap
import xgboost as xgb

# Load and train a model
X, y = shap.datasets.diabetes()
clf = xgb.XGBRegressor().fit(X, y)

st.code('# Load and train a model\n\
X, y = shap.datasets.diabetes()\n\
clf = xgb.XGBRegressor().fit(X, y)')

# Explain model's predictions with SHAP
explainer = shap.Explainer(clf)
shap_values = explainer(X)

st.code("# Explain model's predictions with SHAP\n\
explainer = shap.Explainer(clf)\n\
shap_values = explainer(X)")

st.markdown('Here is a short snippet to create a bee swarm plot of all predictions\
 from the classic Diabetes dataset:')
st.code('# Visualize the predictions explanation\n\
    shap.plots.beeswarm(shap_values, max_display=20)')
# Visualize the predictions explanation
fig, ax = plt.subplots()
fig = shap.plots.beeswarm(shap_values, max_display=20)

st.pyplot(fig = fig)

st.markdown('Try to predict the value of terapia_intensiva in the Italian regions dataset, using SHAP')
X = data_Ita_REGIONI.loc[:,['ricoverati_con_sintomi',
'isolamento_domiciliare',
'nuovi_positivi',
'dimessi_guariti',
'deceduti',
'ingressi_terapia_intensiva',
'totale_casi',
'casi_testati',
'lat',
'long',
'codice_regione'
]]
y = data_Ita_REGIONI.terapia_intensiva

# train a model
clf = xgb.XGBRegressor().fit(X, y)
# Explain model's predictions with SHAP
explainer = shap.Explainer(clf)
shap_values = explainer(X)

# Visualize the predictions explanation
fig, ax = plt.subplots()
fig = shap.plots.bar(shap_values, max_display=12)
st.pyplot(fig)

fig, ax = plt.subplots()
fig = shap.plots.bar(shap_values[0])
st.pyplot(fig)

fig, ax = plt.subplots()
fig = shap.plots.beeswarm(shap_values, max_display=20)
st.pyplot(fig)

fig, ax = plt.subplots()
fig = shap.plots.waterfall(shap_values[0], max_display=20)
st.pyplot(fig)



fig, ax = plt.subplots()
fig = shap.plots.scatter(shap_values[:,"ricoverati_con_sintomi"])
st.pyplot(fig)

fig, ax = plt.subplots()
fig = shap.plots.heatmap(shap_values, max_display=12, feature_values=shap_values.abs.max(0))
fig2, ax2 = plt.subplots()
fig2 = shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1))
st.pyplot(fig)
st.pyplot(fig2)


