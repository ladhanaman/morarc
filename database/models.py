import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    phone_number = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    has_been_welcomed = Column(Boolean, default=False)

    concept_graphs = relationship("ConceptGraph", back_populates="user")

class ConceptGraph(Base):
    __tablename__ = 'concept_graphs'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    domain = Column(String, nullable=False)
    graph_data = Column(Text, nullable=False) # JSON storing nodes and edges
    embedding = Column(Text, nullable=False) # JSON list of floats for RAG
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="concept_graphs")

    def set_embedding(self, emb_list):
        self.embedding = json.dumps(emb_list)

    def get_embedding(self):
        return json.loads(self.embedding) if self.embedding else []

    def set_graph_data(self, data_dict):
        self.graph_data = json.dumps(data_dict)

    def get_graph_data(self):
        return json.loads(self.graph_data) if self.graph_data else {}

class DomainSource(Base):
    __tablename__ = 'domain_sources'
    id = Column(Integer, primary_key=True)
    domain_name = Column(String, unique=True, nullable=False)
    domain_embedding = Column(Text, nullable=False) # Semantic vector of domain name
    trusted_sites = Column(Text, nullable=False) # JSON list of 5-6 verified URLs

    def set_embedding(self, emb_list):
        self.domain_embedding = json.dumps(emb_list)

    def get_embedding(self):
        return json.loads(self.domain_embedding) if self.domain_embedding else []

    def set_sites(self, sites_list):
        self.trusted_sites = json.dumps(sites_list)

    def get_sites(self):
        return json.loads(self.trusted_sites) if self.trusted_sites else []

# Use local SQLite for now. Can be swapped to Turso Postgres/SQLite URL later via env
DB_URL = os.getenv("DATABASE_URL", "sqlite:///morarc.db")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
