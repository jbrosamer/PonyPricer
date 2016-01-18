#! -*- coding: utf-8 -*-

"""
Web Scraper Project
Scrape data from a regularly updated website livingsocial.com and
save to a database (postgres).
Database models part - defines table for storing scraped data.
Direct run will create the table.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

import settings


DeclarativeBase = declarative_base()


def db_connect():
    """Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance.
    """
    return create_engine(URL(**settings.DATABASE))


def create_deals_table(engine):
    """"""
    DeclarativeBase.metadata.create_all(engine)


class HorseAds(DeclarativeBase):
    """Sqlalchemy horse model"""
    __tablename__ = "horseads"

    horse_id = Column('horse_id', Integer, primary_key=True)
    link = Column('link', String, nullable=True)
    breed_id = Column('breed_id', Integer, nullable=True)
    zip_code = Column('zip_code', Integer, nullable=True)
    price = Column('price', Integer, nullable=True)
    color = Column('color', Integer, nullable=True)
    gender_id = Column('gender_id', Integer, nullable=True)
    height = Column('height', Decimal, nullable=True)
    age = Column('age', Integer, nullable=True)
    price = Column('price', Integer, nullable=True)

class Breeds(DeclarativeBase):
	breed_id = Column('breed_id', Integer, primary_key=True)
	breed_name = Column('breed_name', String)
	is_warmblood = Column('is_warmblood', Boolean, default=False)
	is_stock = Column('is_stock', Boolean, default=False)

class Skills(DeclarativeBase):
	skill_id = Column('skill_id', Integer, primary_key=True)
	skill_name = Column('skill_name', String)

class Levels(Dec)


