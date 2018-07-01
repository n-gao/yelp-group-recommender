from sqlalchemy import Integer, Column, ForeignKey, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func
from PIL import Image

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = Column(String(22), primary_key=True)
    name = Column(String(255))
    review_count = Column(Integer)
    yelping_since = Column(DateTime)
    useful = Column(Integer)
    funny = Column(Integer)
    cool = Column(Integer)
    fans = Column(Integer)
    average_stars = Column(Float)
    compliment_hot = Column(Integer)
    compliment_more = Column(Integer)
    compliment_profile = Column(Integer)
    compliment_cute = Column(Integer)
    compliment_list = Column(Integer)
    compliment_note = Column(Integer)
    compliment_plain = Column(Integer)
    compliment_cool = Column(Integer)
    compliment_funny = Column(Integer)
    compliment_writer = Column(Integer)
    compliment_photos = Column(Integer)
    friend_count = Column(Integer)
    elite_years = relationship('EliteYears', back_populates='user')
    friends = relationship('Friend', foreign_keys='Friend.user_id', back_populates='user')
    friended = relationship('Friend', foreign_keys='Friend.friend_id', back_populates='friend')
    reviews = relationship('Review', back_populates='user')
    tips = relationship('Tip', back_populates='user')
    
class Friend(Base):
    __tablename__ = 'friend'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(22), ForeignKey('user.id'))
    user = relationship('User', foreign_keys=[user_id], back_populates='friends')
    friend_id = Column(String(22), ForeignKey('user.id'))
    friend = relationship('User', foreign_keys=[friend_id], back_populates='friended')
    
class EliteYears(Base):
    __tablename__ = 'elite_years'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(22), ForeignKey('user.id'))
    user = relationship('User', back_populates='elite_years')
    year = Column(String(4))
    
class Business(Base):
    __tablename__ = 'business'
    id = Column(String(22), primary_key=True)
    name = Column(String(255))
    neighborhood = Column(String(255))
    address = Column(String(255))
    city = Column(String(255))
    state = Column(String(255))
    postal_code = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    stars = Column(Float)
    review_count = Column(Integer)
    is_open = Column(Boolean)
    attributes = relationship('Attribute', back_populates='business')
    categories = relationship('Category', back_populates='business')
    checkins = relationship('Checkin', back_populates='business')
    hours = relationship('Hours', back_populates='business')
    photos = relationship('Photo', back_populates='business')
    reviews = relationship('Review', back_populates='business')
    tips = relationship('Tip', back_populates='business')
    
class Attribute(Base):
    __tablename__ = 'attribute'
    id = Column(Integer, primary_key=True)
    business_id = Column(String(22), ForeignKey('business.id'))
    business = relationship('Business', back_populates='attributes')
    name = Column(String(255))
    value = Column(Text(65535))
    
class Category(Base):
    __tablename__ = 'category'
    id = Column(Integer, primary_key=True)
    business_id = Column(String(22), ForeignKey('business.id'))
    business = relationship('Business', back_populates='categories')
    category = Column(String(255))
    
class Checkin(Base):
    __tablename__ = 'checkin'
    id = Column(Integer, primary_key=True)
    business_id = Column(String(22), ForeignKey('business.id'))
    business = relationship('Business', back_populates='checkins')
    
class Hours(Base):
    __tablename__ = 'hours'
    id = Column(Integer, primary_key=True)
    business_id = Column(String(22), ForeignKey('business.id'))
    business = relationship('Business', back_populates='hours')
    hours = Column(String(255))  
    
class Photo(Base):
    __tablename__ = 'photo'
    id = Column(Integer, primary_key=True)
    business_id = Column(String(22), ForeignKey('business.id'))
    business = relationship('Business', back_populates='photos')
    caption = Column(String(255))
    label = Column(String(255))
    
    def get_image(self, img_dir='photos/'):
        file = img_dir + self.id + '.jpg'
        return Image.open(file)
    
    
class Review(Base):
    __tablename__ = 'review'
    id = Column(Integer, primary_key=True)
    business_id = Column(String(22), ForeignKey('business.id'))
    business = relationship('Business', back_populates='reviews')
    user_id = Column(String(22), ForeignKey('user.id'))
    user = relationship('User', back_populates='reviews')
    stars = Column(Integer)
    date = Column(DateTime)
    text = Column(Text(65535))
    useful = Column(Integer)
    funny = Column(Integer)
    cool = Column(Integer)
    
class Tip(Base):
    __tablename__ = 'tip'
    id = Column(Integer, primary_key=True)
    business_id = Column(String(22), ForeignKey('business.id'))
    business = relationship('Business', back_populates='tips')
    user_id = Column(String(22), ForeignKey('user.id'))
    user = relationship('User', back_populates='tips')
    date = Column(DateTime)
    text = Column(Text(65535))
    likes = Column(Integer)
