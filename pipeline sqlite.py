#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def try_sqlite():

    import sqlite3
    from sqlite3 import connect

    conn = sqlite3.connect("/Users/fadilamoussaoui/Desktop/Session H22/PSY4016 - Programmation/Projet/Base_de_données_travail.csv")
    tableau = 'MEC'
    conn.execute('''CREATE TABLE IF NOT EXISTS {0} (Sex, Meditation, Empathy.Agreeableness, Disinhibition)'''
             .format(tableau,))
    conn.commit() 

def __definir_les_donnees_(tableau, donnees):
    conn = __connect_db()
    conn.execute('''INSERT INTO {0} VALUES {1}'''.format(tableau, donnees))
    conn.execute("""PRAGMA table_info(MEC)""").fetchall()
    conn.commit()
    conn.close()

