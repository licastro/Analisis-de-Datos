# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 02:16:18 2024

@author: Lucía
"""

#%%

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%%

pais = pd.read_csv('pais.csv')
liga = pd.read_csv('liga.csv')
equipo = pd.read_csv('equipo.csv')
partido = pd.read_csv('partido.csv')
goles = pd.read_csv('goles.csv')
temporada = pd.read_csv('temporada.csv')
plantel = pd.read_csv('plantel.csv')
caracteristicas = pd.read_csv('caracteristicas.csv')
jugador = pd.read_csv('jugador.csv')
equipo_anota_gol = pd.read_csv('equipo_anota_gol.csv')
equipo_forma_plantel = pd.read_csv('equipo_forma_plantel.csv')
jugador_en_plantel = pd.read_csv('jugador_en_plantel.csv')
jugador_hace_gol = pd.read_csv('jugador_hace_gol.csv')
local = pd.read_csv('local.csv')
partido_en_temporada = pd.read_csv('partido_en_temporada.csv')
partido_tiene_goles = pd.read_csv('partido_tiene_goles.csv')
pertenece_a_liga = pd.read_csv('pertenece_a_liga.csv')
pertenece_a_pais = pd.read_csv('pertenece_a_pais.csv')
plantel_de_la_temporada = pd.read_csv('plantel_de_la_temporada.csv')
se_juega_en_liga = pd.read_csv('se_juega_en_liga.csv')
visitante = pd.read_csv('visitante.csv')

#%%

pais_italia = duckdb.sql('''SELECT p.*
                            FROM pais AS p
                            WHERE nombre_pais = 'Italy' ''').df()

#%%

liga_italia = duckdb.sql('''SELECT l.*
                            FROM liga AS l
                            INNER JOIN pertenece_a_pais AS pap ON l.id_liga = pap.id_liga
                            INNER JOIN pais_italia AS pi ON pap.id_pais = pi.id_pais ''').df()

#%%

partidos_italia = duckdb.sql('''SELECT p.*
                                FROM partido AS p
                                INNER JOIN se_juega_en_liga AS sjl ON p.id_partido = sjl.id_partido
                                INNER JOIN pertenece_a_pais AS pap ON sjl.id_liga = pap.id_liga
                                INNER JOIN pais_italia AS pi ON pap.id_pais = pi.id_pais''').df()

#%%

partidos_jugados_por_season = duckdb.sql('''SELECT split_part(id_season, '-', 2) AS season, COUNT(*) AS partidos_jugados
                                            FROM partidos_italia AS pi
                                            INNER JOIN partido_en_temporada AS pt ON pi.id_partido = pt.id_partido
                                            GROUP BY season
                                            ORDER BY partidos_jugados DESC, season ASC;''').df()

#%%

temporada_italia = duckdb.sql('''SELECT t.*
                                 FROM temporada AS t
                                 INNER JOIN pais_italia AS pi ON split_part(t.id_season, '-', 1) = pi.id_pais
                                 WHERE split_part(t.id_season, '-', 2) >= '2012/2013'
                                 AND split_part(t.id_season, '-', 2) <= '2015/2016' ''').df()

#%%

partidos_italia = duckdb.sql('''SELECT DISTINCT pi.*
                                FROM partidos_italia AS pi
                                INNER JOIN partido_en_temporada AS pt ON pi.id_partido = pt.id_partido
                                INNER JOIN temporada_italia AS ti ON pt.id_season = ti.id_season
                                ORDER BY pi.fecha_partido, pi.id_partido ASC;''').df()

#%%

equipos_italia = duckdb.sql('''SELECT DISTINCT e.*
                               FROM equipo AS e
                               INNER JOIN local AS l ON e.id_equipo = l.id_equipo
                               INNER JOIN partidos_italia AS pi ON l.id_partido = pi.id_partido ''').df()

#%%

goles_italia = duckdb.sql('''SELECT DISTINCT g.*
                             FROM goles AS g
                             INNER JOIN partido_tiene_goles AS ptg ON g.id_gol = ptg.id_gol
                             INNER JOIN partidos_italia AS pi ON ptg.id_partido = pi.id_partido''').df()

#%%

plantel_italia = duckdb.sql('''SELECT pl.*
                               FROM plantel AS pl
                               INNER JOIN plantel_de_la_temporada AS plt ON pl.id_plantel = plt.id_plantel
                               INNER JOIN temporada_italia AS ti ON plt.id_season = ti.id_season''').df()

#%%

jugadores_italia = duckdb.sql('''SELECT j.*
                                 FROM jugador AS j
                                 INNER JOIN jugador_en_plantel AS jp ON j.id = jp.id_jugador
                                 INNER JOIN plantel_italia AS pli ON jp.id_plantel = pli.id_plantel''').df()

#%%

caracteristicas_italia = duckdb.sql('''SELECT c.*
                                       FROM caracteristicas AS c
                                       INNER JOIN jugadores_italia AS ji ON c.id_jugador = ji.id
                                       INNER JOIN plantel_italia as pli ON c.id_plantel = pli.id_plantel''').df()

#%%

partidos_ganados = duckdb.sql(''' SELECT ei.nombre_equipo, COUNT(*) AS partidos_ganados
                                  FROM partidos_italia AS pi
                                  INNER JOIN equipos_italia AS ei ON (pi.equipo_local = ei.id_equipo OR pi.equipo_visitante = ei.id_equipo)
                                  WHERE (goles_local > goles_visitante AND pi.equipo_local = ei.id_equipo)
                                  OR (goles_visitante > goles_local AND pi.equipo_visitante = ei.id_equipo)
                                  GROUP BY ei.nombre_equipo
                                  ORDER BY partidos_ganados DESC''').df()

#%%

partidos_perdidos_season = duckdb.sql('''SELECT ei.nombre_equipo, split_part(ti.id_season, '-', 2) AS season, COUNT(*) AS partidos_perdidos
                                         FROM partidos_italia AS pi
                                         INNER JOIN equipos_italia AS ei ON (pi.equipo_local = ei.id_equipo OR pi.equipo_visitante = ei.id_equipo)
                                         INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio
                                         AND pi.fecha_partido <= ti.fecha_fin
                                         WHERE (goles_local < goles_visitante AND pi.equipo_local = ei.id_equipo)
                                         OR (goles_visitante < goles_local AND pi.equipo_visitante = ei.id_equipo)
                                         GROUP BY ei.nombre_equipo, ti.id_season
                                         ORDER BY partidos_perdidos DESC''').df()

#%%

partidos_empatados = duckdb.sql('''SELECT ei.nombre_equipo, COUNT(*) AS partidos_empatados
                                   FROM partidos_italia AS pi
                                   INNER JOIN equipos_italia AS ei ON (pi.equipo_local = ei.id_equipo OR pi.equipo_visitante = ei.id_equipo)
                                   INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio
                                   AND pi.fecha_partido <= ti.fecha_fin
                                   WHERE pi.goles_local = pi.goles_visitante AND ti.id_season = '10257-2015/2016'
                                   GROUP BY ei.nombre_equipo
                                   ORDER BY partidos_empatados DESC;''').df()

#%%

mas_goles_a_favor = duckdb.sql('''SELECT ei.nombre_equipo,
                                         SUM(CASE WHEN pi.equipo_local = ei.id_equipo THEN pi.goles_local
                                             ELSE pi.goles_visitante END) AS goles_a_favor
                                  FROM partidos_italia pi
                                  INNER JOIN equipos_italia ei ON pi.equipo_local = ei.id_equipo
                                  OR pi.equipo_visitante = ei.id_equipo
                                  GROUP BY ei.nombre_equipo
                                  ORDER BY goles_a_favor DESC;''').df()

#%%

mayor_diferencia_goles = duckdb.sql('''SELECT ei.nombre_equipo,
                                              SUM(CASE WHEN pi.equipo_local = ei.id_equipo THEN pi.goles_local - pi.goles_visitante
                                                       ELSE pi.goles_visitante - pi.goles_local END) AS diferencia_goles
                                       FROM partidos_italia pi
                                       INNER JOIN equipos_italia ei ON ei.id_equipo = pi.equipo_local
                                       OR ei.id_equipo = pi.equipo_visitante
                                       GROUP BY ei.nombre_equipo
                                       ORDER BY diferencia_goles DESC;''').df()

#%%

cantidad_jugadores_distintos_plantel = duckdb.sql('''SELECT ei.nombre_equipo, COUNT(DISTINCT ci.id_jugador) AS cantidad_jugadores
                                                     FROM caracteristicas_italia AS ci
                                                     INNER JOIN equipos_italia AS ei ON split_part(ci.id_plantel, '-', 3) = ei.id_equipo
                                                     GROUP BY ei.nombre_equipo
                                                     ORDER BY cantidad_jugadores DESC''').df()
                                                     
equipo_cantidad_temporadas_jugadores = duckdb.sql('''SELECT ei.nombre_equipo, COUNT(id_plantel) AS cantidad_temporadas, cjp.cantidad_jugadores
                                                     FROM plantel_italia AS pi
                                                     INNER JOIN equipos_italia AS ei ON split_part(pi.id_plantel, '-', 3) = ei.id_equipo
                                                     INNER JOIN cantidad_jugadores_distintos_plantel AS cjp ON ei.nombre_equipo = cjp.nombre_equipo
                                                     GROUP BY ei.nombre_equipo, cjp.cantidad_jugadores
                                                     ORDER BY cantidad_temporadas DESC, cjp.cantidad_jugadores DESC''').df()                                                   

#%%

jugador_mayor_cantidad_goles = duckdb.sql('''SELECT DISTINCT ji.nombre, COUNT(DISTINCT gi.id_gol) AS cantidad_goles
                                             FROM goles_italia AS gi
                                             INNER JOIN jugadores_italia AS ji ON gi.jugador_que_anoto = ji.id
                                             WHERE gi.tipo_gol NOT IN ('o', 'dg', 'npm')
                                             GROUP BY ji.nombre
                                             ORDER BY cantidad_goles DESC''').df()

#%%

partidos_ganados_season = duckdb.sql('''SELECT CASE WHEN pi.goles_local > pi.goles_visitante THEN pi.equipo_local
                                                    WHEN pi.goles_visitante > pi.goles_local THEN pi.equipo_visitante
                                                    END AS id_equipo,
                                               split_part(ti.id_season, '-', 2) AS season, COUNT(*) AS partidos_ganados
                                        FROM partidos_italia AS pi
                                        INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio
                                        AND pi.fecha_partido <= ti.fecha_fin
                                        GROUP BY id_equipo, ti.id_season
                                        ORDER BY partidos_ganados DESC''').df()

jugadores_mas_partidos_ganados = duckdb.sql('''SELECT DISTINCT ji.nombre, SUM(DISTINCT pgs.partidos_ganados) AS partidos_ganados
                                               FROM partidos_ganados_season AS pgs
                                               INNER JOIN caracteristicas_italia AS ci ON pgs.season = split_part(ci.id_plantel, '-', 2)
                                               AND pgs.id_equipo = split_part(ci.id_plantel, '-', 3)
                                               INNER JOIN jugadores_italia AS ji ON ci.id_jugador = ji.id
                                               GROUP BY ji.nombre
                                               ORDER BY partidos_ganados DESC''').df()

#%%

jugador_en_mas_equipos = duckdb.sql('''SELECT DISTINCT ji.nombre, COUNT(DISTINCT split_part(id_plantel, '-', 3)) AS cantidad_equipos
                                       FROM caracteristicas_italia AS ci
                                       INNER JOIN jugadores_italia AS ji ON ci.id_jugador = ji.id
                                       GROUP BY ji.nombre
                                       ORDER BY cantidad_equipos DESC''').df()

#%%

menor_variacion_potencia = duckdb.sql('''SELECT ji.nombre, ABS(MAX(potencia) - MIN(potencia)) AS variacion_potencia
                                         FROM caracteristicas_italia AS ci
                                         INNER JOIN jugadores_italia AS ji ON ci.id_jugador = ji.id
                                         GROUP BY ji.nombre
                                         ORDER BY variacion_potencia ASC''').df()
                                         
#%%

ganadores_partidos_italia = duckdb.sql('''SELECT id_partido,
                                                 CASE WHEN goles_local > goles_visitante THEN equipo_local
                                                      ELSE equipo_visitante END AS equipo_ganador
                                          FROM partidos_italia
                                          WHERE goles_local != goles_visitante ''').df()

goles_id_partido = duckdb.sql('''SELECT pi.id_partido, pi.equipo_local, pi.equipo_visitante, pi.fecha_partido, gi.*
                                 FROM goles_italia AS gi
                                 INNER JOIN partidos_italia AS pi ON pi.fecha_partido = gi.fecha_gol
                                 AND (pi.equipo_local = gi.equipo_que_anoto OR pi.equipo_visitante = gi.equipo_que_anoto)
                                 WHERE gi.tipo_gol NOT IN ('o', 'dg', 'npm') ''').df()

primer_equipo_que_anoto = duckdb.sql('''SELECT gip.id_partido, gip.equipo_local, gip.equipo_visitante, gip.id_gol, gip.equipo_que_anoto,
                                               gip.tiempo, gip.ronda, gip.fecha_partido
                                        FROM goles_id_partido AS gip
                                        WHERE gip.tiempo = (SELECT MIN(gip2.tiempo)
                                                            FROM goles_id_partido AS gip2
                                                            WHERE gip2.id_partido = gip.id_partido)''').df()

remontaron_partidos = duckdb.sql('''SELECT pea.id_partido, pea.id_gol, gpi.equipo_ganador, pea.equipo_que_anoto AS empezo_ganando, pea.fecha_partido
                                    FROM primer_equipo_que_anoto AS pea
                                          LEFT JOIN ganadores_partidos_italia AS gpi ON pea.id_partido = gpi.id_partido
                                          WHERE gpi.equipo_ganador != empezo_ganando''').df()

cantidad_partidos_remontados = duckdb.sql('''SELECT split_part(ti.id_season, '-', 2) AS season, ei.nombre_equipo,
                                                    COUNT(DISTINCT rp.id_partido) AS partidos_remontados
                                             FROM remontaron_partidos AS rp
                                             INNER JOIN partido_en_temporada AS pt ON rp.id_partido = pt.id_partido
                                             INNER JOIN temporada_italia AS ti ON pt.id_season = ti.id_season
                                             INNER JOIN equipos_italia AS ei ON rp.equipo_ganador = ei.id_equipo
                                             GROUP BY ti.id_season, ei.nombre_equipo
                                             ORDER BY partidos_remontados DESC''').df()
                                             
#%%

mas_goles_a_favor_season = duckdb.sql('''SELECT ei.nombre_equipo, split_part(ti.id_season, '-', 2) AS season,
                                                SUM(CASE WHEN pi.equipo_local = ei.id_equipo THEN pi.goles_local
                                                         ELSE pi.goles_visitante END) AS goles_a_favor
                                        FROM partidos_italia pi
                                        INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio AND pi.fecha_partido <= ti.fecha_fin
                                        INNER JOIN equipos_italia ei ON pi.equipo_local = ei.id_equipo OR pi.equipo_visitante = ei.id_equipo
                                        GROUP BY ei.nombre_equipo, season, ei.id_equipo
                                        ORDER BY ei.id_equipo, season ASC''').df()                                         

mas_goles_en_contra_season = duckdb.sql('''SELECT ei.nombre_equipo,split_part(ti.id_season, '-', 2) AS season,
                                                SUM(CASE WHEN pi.equipo_local = ei.id_equipo THEN pi.goles_visitante
                                                         ELSE pi.goles_local END) AS goles_en_contra
                                         FROM partidos_italia pi
                                         INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio AND pi.fecha_partido <= ti.fecha_fin
                                         INNER JOIN equipos_italia ei ON pi.equipo_local = ei.id_equipo OR pi.equipo_visitante = ei.id_equipo
                                         GROUP BY ei.nombre_equipo, season, ei.id_equipo
                                         ORDER BY ei.id_equipo, season ASC''').df()
                                         
fig, ax = plt.subplots(figsize=(8,6))
for i, g in mas_goles_en_contra_season.groupby('nombre_equipo'):
    g.plot(x='season', y='goles_en_contra', ax=ax, label=str(i))

ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_title('Cantidad de goles en contra por equipo, por temporada')
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
for i, g in mas_goles_a_favor_season.groupby('nombre_equipo'):
    g.plot(x='season', y='goles_a_favor', ax=ax, label=str(i))

ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_title('Cantidad de goles a favor por equipo, por temporada')
plt.show()

#%%

goles_a_favor_promedio= mas_goles_a_favor_season.groupby(by=['nombre_equipo','season'], as_index=False).mean("mas_goles_a_favor_season")                                       

fig, ax = plt.subplots(figsize=(8,6))
for i, g in goles_a_favor_promedio.groupby('nombre_equipo'):
    g.plot(x='season', y='goles_a_favor', ax=ax, label=str(i))
ax.set_title('Promedio de goles por equipo, por temporada')
ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

#%%

goles_local_season = duckdb.sql('''SELECT pi.equipo_local AS id, split_part(ti.id_season, '-', 2) AS season,
                                          SUM(pi.goles_local) AS cantidad_goles_local
                                    FROM partidos_italia AS pi
                                    INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio AND pi.fecha_partido <= ti.fecha_fin
                                    GROUP BY pi.equipo_local, ti.id_season
                                    ORDER BY pi.equipo_local, season ASC;''').df()  
                                    
goles_visitante_season = duckdb.sql('''SELECT pi.equipo_visitante AS id, split_part(ti.id_season, '-', 2) AS season,
                                              SUM(pi.goles_visitante) AS cantidad_goles_visitante
                                    FROM partidos_italia AS pi
                                    INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio AND pi.fecha_partido <= ti.fecha_fin
                                    GROUP BY pi.equipo_visitante, ti.id_season
                                    ORDER BY pi.equipo_visitante, season ASC''').df()     
                                    
diferencia_goles_local_visitante = duckdb.sql('''SELECT ei.nombre_equipo, SUM(gls.cantidad_goles_local - gvs.cantidad_goles_visitante) AS diferencia_goles,
                                                        gls.season
                                                 FROM goles_local_season AS gls
                                                 INNER JOIN goles_visitante_season AS gvs ON gls.id = gvs.id
                                                 AND gls.season = gvs.season
                                                 INNER JOIN equipos_italia AS ei ON gls.id = ei.id_equipo
                                                 GROUP BY gls.season, ei.nombre_equipo, ei.id_equipo
                                                 ORDER BY ei.id_equipo, gls.season ASC;''').df()

fig, ax = plt.subplots(figsize=(8,6))
for i, g in diferencia_goles_local_visitante.groupby('nombre_equipo'):
    g.plot(x='season', y='diferencia_goles', ax=ax, label=str(i))
ax.set_title('Diferencia goles como local - visitante, por equipo, por temporada')
ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()
                                             
#%%

cantidad_goles_favorables_convertidos = duckdb.sql('''SELECT gi.equipo_que_anoto AS id_equipo,
                                                             split_part(ti.id_season, '-', 2) AS season,
                                                             COUNT(DISTINCT gi.id_gol) AS cantidad_goles
                                                      FROM goles_italia AS gi
                                                      INNER JOIN temporada_italia AS ti ON gi.fecha_gol >= ti.fecha_inicio
                                                      AND gi.fecha_gol <= ti.fecha_fin
                                                      WHERE gi.tipo_gol NOT IN ('o', 'dg', 'npm')
                                                      GROUP BY id_equipo, season''').df()

promedio_overall_rating_plantel = duckdb.sql('''SELECT split_part(pi.id_plantel, '-', 2) AS season,
                                                        split_part(pi.id_plantel, '-', 3) AS id_equipo,
                                                        SUM(ci.overall_rating) / COUNT(*) AS promedio_overall_rating
                                                FROM plantel_italia AS pi
                                                INNER JOIN caracteristicas_italia AS ci ON pi.id_plantel = ci.id_plantel
                                                GROUP BY season, id_equipo''').df()

golesVSatributos = duckdb.sql('''SELECT ppr.*, cgi.*, ei.*
                                 FROM cantidad_goles_favorables_convertidos AS cgi
                                 INNER JOIN promedio_overall_rating_plantel AS ppr ON cgi.id_equipo = ppr.id_equipo
                                 INNER JOIN equipos_italia AS ei ON cgi.id_equipo = ei.id_equipo
                                 AND cgi.season = ppr.season''').df()

plt.figure(figsize=(24, 16))

sns.scatterplot(data=golesVSatributos,
                x='promedio_overall_rating',
                y='cantidad_goles',
                hue='season',
                palette='bright',
                alpha=0.7)

for i in range(golesVSatributos.shape[0]):
    plt.annotate(golesVSatributos['nombre_equipo'].iloc[i],
                 (golesVSatributos['promedio_overall_rating'].iloc[i],
                  golesVSatributos['cantidad_goles'].iloc[i]),
                 textcoords="offset points",
                 xytext=(0,5),
                 ha='center')

plt.title('Promedio Overall Rating vs Goles Convertidos')
plt.xlabel('Promedio Overall Rating')
plt.ylabel('Cantidad de Goles Convertidos')
plt.legend(title='Temporada')
plt.grid()
plt.show()

#%%

cantidad_goles_contra = duckdb.sql('''SELECT gi.equipo_que_anoto AS id_equipo,
                                             split_part(ti.id_season, '-', 2) AS season,
                                             COUNT(DISTINCT gi.id_gol) AS cantidad_goles_contra
                                      FROM goles_italia AS gi
                                      INNER JOIN temporada_italia AS ti ON gi.fecha_gol >= ti.fecha_inicio
                                      AND gi.fecha_gol <= ti.fecha_fin
                                      WHERE gi.tipo_gol IN ('o', 'dg')
                                      GROUP BY gi.equipo_que_anoto, season''').df()

golescontraVSatributos = duckdb.sql('''SELECT ppr.*, cgc.*, ei.*
                                       FROM cantidad_goles_contra AS cgc
                                       INNER JOIN promedio_overall_rating_plantel AS ppr ON cgc.id_equipo = ppr.id_equipo
                                       INNER JOIN equipos_italia AS ei ON cgc.id_equipo = ei.id_equipo
                                       AND cgc.season = ppr.season''').df()

plt.figure(figsize=(24, 16))

sns.scatterplot(data=golescontraVSatributos,
                x='promedio_overall_rating',
                y='cantidad_goles_contra',
                hue='season',
                palette='bright',
                alpha=0.7)

for i in range(golescontraVSatributos.shape[0]):
    plt.annotate(golescontraVSatributos['nombre_equipo'].iloc[i],
                 (golescontraVSatributos['promedio_overall_rating'].iloc[i],
                  golescontraVSatributos['cantidad_goles_contra'].iloc[i]),
                 textcoords="offset points",
                 xytext=(0, 5),
                 ha='center')

plt.title('Promedio Overall Rating vs Goles Convertidos')
plt.xlabel('Promedio Overall Rating')
plt.ylabel('Cantidad de Goles Convertidos')
plt.legend(title='Temporada')
plt.grid()
plt.show()

#%%

goles_por_minuto = duckdb.sql('''SELECT tiempo AS minuto, COUNT(id_gol) AS cantidad_goles
                                 FROM goles_italia
                                 GROUP BY tiempo ORDER BY minuto''').df()

goles_por_minuto['minuto'] = goles_por_minuto['minuto'].astype(int)

plt.figure(figsize=(12, 6))
sns.barplot(data=goles_por_minuto, x='minuto', y='cantidad_goles', hue='minuto', palette='viridis', legend=False)
plt.title('Goles Convertidos por Minuto')
plt.xlabel('Minuto del Partido')
plt.ylabel('Cantidad de Goles')
plt.xticks(rotation=90)  
plt.show()

#%%

goles_por_equipo_minuto = duckdb.sql('''SELECT gi.tiempo AS minuto, gi.ronda AS jornada, COUNT(gi.id_gol) AS cantidad_goles
                                        FROM goles_italia AS gi
                                        GROUP BY minuto, gi.equipo_que_anoto, gi.ronda''').df()

goles_por_equipo_minuto['minuto'] = goles_por_equipo_minuto['minuto'].astype(int)

def clasificar_jornada(jornada):
    if 1 <= jornada <= 8:
        return 'Primera'
    elif 9 <= jornada <= 19:
        return 'Segunda'
    elif 20 <= jornada <= 30:
        return 'Tercera'
    elif 31 <= jornada <= 38:
        return 'Última'

goles_por_equipo_minuto['etapa_jornada'] = goles_por_equipo_minuto['jornada'].apply(clasificar_jornada)

goles_por_equipo_minuto['etapa_jornada'] = pd.Categorical(goles_por_equipo_minuto['etapa_jornada'],
                                                       categories=['Primera', 'Segunda', 'Tercera', 'Última'],
                                                       ordered=True)

heatmap_data = goles_por_equipo_minuto.pivot_table(index='minuto',
                                                    columns='etapa_jornada',
                                                    values='cantidad_goles',
                                                    fill_value=0)

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, cbar=False)
plt.title('Goles Convertidos por Etapa de Jornadas y Minuto')
plt.xlabel('Etapa de Jornadas')
plt.ylabel('Minutos de Partido')
plt.show()

#%%

goles_tipo = duckdb.sql('''SELECT ei.nombre_equipo, 
                                   gi.tipo_gol, 
                                   COUNT(gi.id_gol) AS cantidad_goles
                            FROM goles_italia AS gi
                            INNER JOIN equipos_italia AS ei ON gi.equipo_que_anoto = ei.id_equipo
                            GROUP BY nombre_equipo, tipo_gol''').df()
                            
goles_tipo['tipo_gol'] = goles_tipo['tipo_gol'].replace({
    'n': 'Normal',
    'p': 'Penal',
    'rp': 'Rebote de penal',
    'npm': 'Anulado',
    'o': 'Gol en contra',
    'dg': 'Gol en contra por defensor',
})                           

orden_tipos_gol = ['Normal', 'Penal', 'Rebote de penal', 'Anulado', 'Gol en contra', 'Gol en contra por defensor']

goles_tipo['tipo_gol'] = pd.Categorical(goles_tipo['tipo_gol'], categories=orden_tipos_gol, ordered=True)

goles_pivot = goles_tipo.pivot(index='nombre_equipo', columns='tipo_gol', values='cantidad_goles').fillna(0)

goles_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Distribución de Tipos de Goles por Equipo')
plt.xlabel('Equipos')
plt.ylabel('Cantidad de Goles')
plt.legend(title='Tipo de Gol')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

goles_pivot_normalizado = goles_pivot.div(goles_pivot.sum(axis=1), axis=0)

goles_pivot_normalizado.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Distribución Normalizada de Tipos de Goles por Equipo')
plt.xlabel('Equipos')
plt.ylabel('Proporción de Goles')
plt.legend(title='Tipo de Gol')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

goles_local_visitante = duckdb.sql('''SELECT pi.equipo_local AS id, split_part(ti.id_season, '-', 2) AS season,
                                             SUM(pi.goles_local) AS cantidad_goles, 'Local' AS tipo_gol
                                    FROM partidos_italia AS pi
                                    INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio AND pi.fecha_partido <= ti.fecha_fin
                                    GROUP BY pi.equipo_local, ti.id_season
                                    UNION ALL 
                                    SELECT pi.equipo_visitante AS id, split_part(ti.id_season, '-', 2) AS season,
                                           SUM(pi.goles_visitante) AS cantidad_goles, 'Visitante' AS tipo_gol
                                    FROM partidos_italia AS pi
                                    INNER JOIN temporada_italia AS ti ON pi.fecha_partido >= ti.fecha_inicio AND pi.fecha_partido <= ti.fecha_fin
                                    GROUP BY pi.equipo_visitante, ti.id_season''').df()  

sns.boxplot(data=goles_local_visitante, x='tipo_gol', y='cantidad_goles', palette='muted')

plt.title('Distribución de Goles Anotados: Local vs. Visitante')
plt.xlabel('Tipo de Goles')
plt.ylabel('Cantidad de Goles Anotados')
plt.show()

#%%

plt.figure(figsize=(12, 18))  

for idx, i in enumerate([5, 20, 35], start=1): 
    planteles_mas_goleadores = duckdb.sql('''SELECT *
                                               FROM cantidad_goles_favorables_convertidos
                                               ORDER BY cantidad_goles DESC
                                               LIMIT ?''', params=(i,)).df()

    mejores_goles_local_visitante = duckdb.sql('''SELECT glv.id, glv.cantidad_goles, glv.tipo_gol
                                                    FROM goles_local_visitante AS glv
                                                    INNER JOIN planteles_mas_goleadores AS pg ON glv.id = pg.id_equipo
                                                    AND glv.season = pg.season''').df()      

    planteles_menos_goleadores = duckdb.sql('''SELECT *
                                                FROM cantidad_goles_favorables_convertidos
                                                ORDER BY cantidad_goles 
                                                LIMIT ?''', params=(i,)).df()                    

    peores_goles_local_visitante = duckdb.sql('''SELECT glv.id, glv.cantidad_goles, glv.tipo_gol
                                                   FROM goles_local_visitante AS glv
                                                   INNER JOIN planteles_menos_goleadores AS pg ON glv.id = pg.id_equipo
                                                   AND glv.season = pg.season''').df()      

    plt.subplot(3, 2, idx * 2 - 1) 
    sns.boxplot(data=mejores_goles_local_visitante, x='tipo_gol', y='cantidad_goles', palette='muted')
    plt.title(f'Mejores {i} Planteles')
    plt.xlabel('Tipo de Goles')
    plt.ylabel('Cantidad de Goles Anotados')

    plt.subplot(3, 2, idx * 2)  
    sns.boxplot(data=peores_goles_local_visitante, x='tipo_gol', y='cantidad_goles', palette='muted')
    plt.title(f'Peores {i} Planteles')
    plt.xlabel('Tipo de Goles')
    plt.ylabel('Cantidad de Goles Anotados')

plt.tight_layout()
plt.show()
