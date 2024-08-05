import React, { useContext } from 'react';
import Container from '../Container/Container';
import { ChosenContext } from '../Context/ChosenContext';
import { PredictionContext } from '../Context/PredictionContext';
import { PiListBulletsBold } from "react-icons/pi";
import Slider from 'react-slick';
import img1 from '../../public/pexels-pixabay-139398.jpg'
import img2 from '../../public/pexels-negativespace-48604.jpg'

import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import { FaArrowLeft, FaArrowRight } from 'react-icons/fa';
// import './custom-carousel.css';
const NextArrow = (props) => {
  const { onClick } = props;
  return (
    <div
      className="absolute top-1/4 right-0 transform -translate-y-1/2 bg-gray-100 rounded-full p-2 cursor-pointer z-10"
      onClick={onClick}
    >
      <FaArrowRight />
    </div>
  );
};

const PrevArrow = (props) => {
  const { onClick } = props;
  return (
    <div
      className="absolute top-1/4 left-0 transform -translate-y-1/2 bg-gray-100 rounded-full p-2 cursor-pointer z-10"
      onClick={onClick}
    >
      <FaArrowLeft />
    </div>
  );
};
const DiseaseDescriptionCard = ({ image, name, description }) => (
  <div className=' h-[250px] text-black rounded-xl p-4'>
    <div className='h-56 w-full flex justify-center items-end '>
      <img src={image} alt={name} className='h-full w-full' />
    </div>
    <div className=''>
      <div className='flex items-center justify-center'>
      <p className='sm:text-xl  text-md font-semibold text-gray-700 pt-3 pb-1 '>Disease Predicted: <span className='text-teal-500'>{name}</span></p>
      </div>
      <p className='text-sm sm:text-base'>{description}</p >
    </div>
  </div>
);

const DiseasePrecautionCard = ({ image, name, precautions }) => (
  <div className='h-[250px] text-gray-800 rounded-xl p-4'>
    <div className='h-56 w-full flex justify-center items-end'>
      <img src={image} alt={name} className='h-full w-full' />
    </div>
    <div>
    <div className='flex items-center justify-center'>
      <p className='sm:text-xl  text-md font-semibold text-gray-700 pt-3 pb-1 '> Preacution For: <span className='text-teal-500'>{name}</span></p>
      </div>
      <ul>
        {precautions.map((precaution, index) => (
          <li key={index} className='flex items-center'>
            <PiListBulletsBold className='pr-3' />
            {precaution}
          </li>
        ))}
      </ul>
    </div>
  </div>
);


const Disease = () => {
  const { prediction, description, precaution } = useContext(PredictionContext);
  const { chosen, setChosen } = useContext(ChosenContext);

  const disease = {
    image: img2,
    name: prediction,
    description: description,
    precautions: precaution,
  };

  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    nextArrow: <NextArrow />,
    prevArrow: <PrevArrow />,
  };

  return (
    <Container>
      <div className='max-w-md max-h-10 mx-auto'>
        {prediction ? (
          <Slider {...settings}>
            <div key="description-card">
              <DiseaseDescriptionCard
                image={disease.image}
                name={disease.name}
                description={disease.description}
              />
            </div>
            <div key="precaution-card">
              <DiseasePrecautionCard
                image={img1}
                name={disease.name}
                precautions={disease.precautions}
              />
            </div>
          </Slider>
        ) : (
          <div>
            <p>No Disease available. Please go back and enter symptoms.</p>
          </div>
        )}
      </div>
    </Container>
  );
};

export default Disease;
