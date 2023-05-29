from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func

# Create the database engine
engine = create_engine('sqlite:///drone_definitions2.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

# Define a model class for all tables


class EntityType(Base):
    __tablename__ = 'entity_types'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    term = relationship("Terms", back_populates="entity_types")
    # Add more columns as needed

    def __repr__(self):
        return f'<EntityType(id={self.id}, name={self.name}, description={self.description})>'


class Term(Base):
    __tablename__ = 'terms'

    id = Column(Integer, primary_key=True)
    term = Column(String, nullable=False)
    entity_type_id = Column(Integer, ForeignKey(
        'entity_types.id'), nullable=False)
    entity_type = relationship("EntityType", back_populates="terms")
    prompt = relationship("Prompt", back_populates="terms")
    __table_args__ = (UniqueConstraint('term', 'entity_type_id'),)


class PromptTemplate(Base):
    __tablename__ = 'prompt_templates'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    template = Column(String, nullable=False)
    description = Column(String)


class Prompt(Base):
    __tablename__ = 'prompts'

    id = Column(Integer, primary_key=True)
    prompt = Column(String, nullable=False)
    term_id = Column(Integer, ForeignKey('terms.id'), nullable=False)
    term = relationship("Term", back_populates="prompts")
    template_id = Column(Integer, ForeignKey(
        'prompt_templates.id'), nullable=False)


class Model(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    company = Column(String)
    version = Column(String)
    size = Column(String)


class Attempt(Base):
    __tablename__ = 'attempts'

    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.id'), nullable=False)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    counter = Column(Integer, nullable=False)


class Definition(Base):
    __tablename__ = 'definitions'

    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.id'), nullable=False)
    term = relationship("Term", back_populates="definitions")
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    model_name = relationship("Model", back_populates="definitions")
    prompt_id = Column(Integer, ForeignKey('prompts.id'), nullable=False)
    prompt = relationship("Prompts", back_populates="definitions")
    counter = Column(Integer, nullable=False)
    cosine_distance = Column(Float, nullable=False)
    euclidean_distance = Column(Float, nullable=False)
    manhattan_distance = Column(Float, nullable=False)
    sum_distance = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def table_to_model(table_name):
    if table_name == 'entity_types':
        return EntityType()
    elif table_name == 'terms':
        return Term()


def main():
    # Create the table if it doesn't exist
    print("creating database...")
    Base.metadata.create_all(engine)
    print("database created successfully...")

# CRUD operations


def create_model(table_name, data):
    model = table_to_model(table_name)
    insert_query = model(data)
    session.add(insert_query)
    session.commit()
    return model


# def read_model(model_id):
#     model = session.query(MyModel).filter_by(id=model_id).first()
#     return model


# def update_model(model_id, new_name):
#     model = session.query(MyModel).filter_by(id=model_id).first()
#     if model:
#         model.name = new_name
#         session.commit()
#         return True
#     return False


# def delete_model(model_id):
#     model = session.query(MyModel).filter_by(id=model_id).first()
#     if model:
#         session.delete(model)
#         session.commit()
#         return True
#     return False

if __name__ == "__main__":
    main()
