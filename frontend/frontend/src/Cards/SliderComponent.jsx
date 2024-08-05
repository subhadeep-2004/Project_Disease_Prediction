import React from 'react';
import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import DiseaseDescriptionCard from './DiseaseDescriptionCard';
import DiseasePrecautionCard from './DiseasePrecautionCard';

const SliderComponent = ({ diseases }) => {
  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
  };

  return (
    <Slider {...settings}>
      {diseases.map((disease, index) => (
        <div key={index}>
          <DiseaseDescriptionCard
            image={disease.image}
            name={disease.name}
            description={disease.description}
          />
          <DiseasePrecautionCard
            image={disease.image}
            name={disease.name}
            precautions={disease.precautions}
          />
        </div>
      ))}
    </Slider>
  );
};

export default SliderComponent;
