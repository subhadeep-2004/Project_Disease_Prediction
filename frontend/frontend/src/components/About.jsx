import React from 'react'
import Container from '../Container/Container.jsx'
import Navbar from './Navbar.jsx'

const About = () => {
  return (
   <>

    <Container>

    <div className='max-w-md max-h-15 mx-auto p-4'>

    <p className='items-center'>
    Welcome to this website, our reliable online 
    tool for symptom analysis and medical prognosis. 
    You can input your symptoms and receive an instant,
     <span className='text-teal-700'> ML-driven </span>prognosis tailored to your specific condition.
     Our user-friendly interface ensures a seamless experience, making health self-assessment accessible to everyone. <br /> 
    <span className='text-red-700 font-semibold'> Always Remember, this is not a substitute for consulting with a healthcare professional. It should be used as an additional resource, not as a sole source of medical guidance. </span>
    </p>
    <p></p>
    </div>

    <div className=' flex items-center justify-center text-teal-500 text-lg font-semibold'>
    "Stay strong, live long"
    </div>
 


    </Container>
   
   
   
   
   </>
  )
}

export default About
