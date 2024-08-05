import React, { useContext, useEffect, useState } from 'react'
import Navbar from './Navbar'
import doctors from "../../public/Chat bot-amico.svg"
import Container from '../Container/Container'
import Symptom from '../shared/Symptom'
import { BiChevronDown } from "react-icons/bi";
import { AiOutlineSearch } from "react-icons/ai";
import { IoIosCloseCircleOutline } from "react-icons/io";
import { IoIosAddCircleOutline } from "react-icons/io";
import { Allsymptoms } from '../shared/AllSymptom'
import img1 from '../assets/stethoscope.jpg'
import Card from '../shared/Card'
import { Link  ,useNavigate } from 'react-router-dom'
import { useAccess } from '../AcessProvider/AcessProvider'
import { ChosenContext } from '../Context/ChosenContext'
import axios from 'axios';
import { PredictionContext } from '../Context/PredictionContext';

import { ClipLoader } from 'react-spinners'



// import AnimatedGif from '../assets/AnimatedGif'
const Selector = () => {


  const { setHasAccess } = useAccess();
  const navigate = useNavigate();


  

  
  const{chosen , setChosen} = useContext(ChosenContext);

  const { setPrediction } = useContext(PredictionContext);
  const{ setDescription } = useContext(PredictionContext);
  const{setPrecaution} = useContext(PredictionContext)
  const [symptoms, setSymptoms] = useState(null)

  const [inputValue, setInputValue] = useState("")

  const[selected,setSelected] = useState("")
  // Load all the symptoms when the page render
  const [ open, setOpen ] = useState(false) 

  const [loading, setLoading] = useState(false);

  // const [prediction, setPrediction] = useState(null);

  // const[chosen, setChosen] = useState([]);


  console.log(chosen);

  useEffect(() => {
    const loadSymptoms = () => {
      setSymptoms(Allsymptoms);
    }
    // const loadChosen =()=>{
    //   setChosen(chosen)
    // }
    // loadChosen()
    loadSymptoms();

  }, []);

  




  const handleAdd=() =>{
    if(selected){

      if(!chosen.includes(selected)){

        setChosen([...chosen, selected]);
        setSelected("")
        
      }else{
        console.log("Already Symptom is selected");
      }
    }else{
      console.log("No value selected");
    }

  }

  const handleSubmit = async (e)=>{
    e.preventDefault()
    console.log("wd");
   
    console.log(chosen);


  
    setLoading(true);


      
    // } catch (error) {
    //   console.error('Error:', error);
    // }

    try {
      const response = await axios.post('/disease', {
        given_symps: chosen,
      },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    
    
    
    );
      console.log(response.data.prediction);
      console.log(response.data.description);
      setPrediction(response.data.prediction);
      setDescription(response.data.description);
      setPrecaution(response.data.precaution);

      // console.log(description);
      // console.log(prediction);
    } catch (error) {
      console.error('Error making prediction:', error);
    }

    navigate("/disease")
    setHasAccess(true)
  




    return chosen

   
    
  }

  return (

    <form  onSubmit={handleSubmit} action='POST'> 
    <div className='w-72 font-medium h-80'>
      <div className='flex justify-center gap-5 items-center'>
      <div
      onClick={()=>setOpen(!open)}
      
      className={`bg-teal-500 cursor-pointer  w-full p-2 flex items-center justify-between rounded text-white ${!selected && "text-gray-800"} `}>
       { selected ?
       selected.length > 25?
        selected.substring(0,25)+"..." 
       :selected
       :"Select a symptom" }
        
        {
          open ? <IoIosCloseCircleOutline /> :<BiChevronDown />
        }
      </div>

      <IoIosAddCircleOutline onClick={handleAdd} className='' size={30}/>
      {/* <button type="button" class="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-3 py-2.5 me-2 mb-2 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800">Default</button> */}
        

      </div>

      

      <ul className={`bg-white mt-2 overflow-y-auto ${open ? "max-h-60": "max-h-0"  } `}>
        <div className='flex items-center px-2 sticky top-0 bg-white z-20'>
        <AiOutlineSearch size={18} className='text-gray-700'/>
        <input type="text" name='symptoms_list' placeholder="Enter your symptom"
        value={inputValue}
         className="placeholder:text-gray-500 z-20 p-2 w-full outline-none" 
         onChange={(e)=>setInputValue(e.target.value.toLowerCase() ) }
         />
        </div>
        {
          symptoms?.map(symptom => {

            return (
              <li
              className={`p-2 text-sm  cursor-pointer
              ${symptom.toLowerCase() ===selected.toLowerCase() ? "hover: bg-teal-400" : "hover:bg-teal-200" }  
              ${symptom.toLowerCase() ===selected.toLowerCase() && "bg-teal-400" && "text-white"}  
                
              ${symptom.toLowerCase().startsWith(inputValue)?'block':'hidden'} `}
              onClick={()=>{
                if(symptom.toLowerCase() !== selected.toLowerCase() ){

                  setSelected(symptom);
                  setOpen(false)
                  setInputValue("")

                }else{

                }
              }}

              
              >

                {symptom}

              </li>
            )
          })
        }

      </ul>



      {
         chosen.length==0 && !open && <div className=' p-3'> 
            <div className='text-sm text-gray-700'>
            Most Searches
            </div>

            <div className='flex flex-wrap m-2'> 

             <div className='flex items-center'>
            <div className='bg-gray-200 text-xs rounded-full  p-2 m-1 inline-block border-2 border-gray-600'>
               {"skin-rash"}
            </div>

            </div> 

            
            <div className='flex items-center'>
            <div className='bg-gray-200 text-xs rounded-full  p-2 m-1 inline-block border-2 border-gray-600'>
               {"itching"}
            </div>

            </div> 


            
            <div className='flex items-center'>
            <div className='bg-gray-200 text-xs rounded-full  p-2 m-1 inline-block border-2 border-gray-600'>
               {"coughing"}
            </div>

            </div> 

             
            <div className='flex items-center'>
            <div className='bg-gray-200 text-xs rounded-full  p-2 m-1 inline-block border-2 border-gray-600'>
               {"shivering"}
            </div>

            </div> 


             
            <div className='flex items-center'>
            <div className='bg-gray-200 text-xs rounded-full  p-2 m-1 inline-block border-2 border-gray-600'>
               {"anxiety"}
            </div>

            </div> 



             
            <div className='flex items-center'>
            <div className='bg-gray-200 text-xs rounded-full  p-2 m-1 inline-block border-2 border-gray-600'>
               {" restlessness"}
            </div>

            </div> 




            </div>
          
          
          </div>
        }







        {
          chosen.length>0 &&<div className='flex flex-wrap m-5'>
          {chosen.map((symptom, index) => (
            <Symptom key={index} name={symptom} chosen={chosen} setChosen={setChosen}/>
          ))}
        </div>
        }

        {
          chosen.length>0 && 
        
             <div className='w-full flex justify-center'>
               <button type="submit" class="text-white bg-teal-700 hover:bg-teal-800 focus:ring-4 focus:ring-teal-300 font-medium rounded-lg text-sm px-3 py-2.5 me-2 mb-2 dark:bg-teal-600 dark:hover:bg-teal-700 focus:outline-none dark:focus:ring-teal-800">
                
                
               {loading==true?  (<ClipLoader size={15} color={"white"} loading={loading}  className='mx-3'/>):
                ( <div>Submit</div> )
               }

                 
                  </button>

             </div>
          
        
    
        
        }

        {/* {
           chosen.length>0 && <input type="text" name='symptoms_list'  value={chosen} />
        } */}



    </div>

    </form>
  )
}


// const RightPart=()=>{

//   return (
//     <div className='justify-center w-full items-center overflow-auto'>
//       Hold On
//       <Card imgSrc={img1}>

//       <h1>Disease</h1>
//       <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Labore vero iusto totam natus voluptatem beatae corporis corrupti sint aliquid impedit? Quam optio quas consectetur quidem, rerum dicta itaque similique repudiandae!</p>
//       </Card>
//     </div>
//   )
// }

const Detail =()=>{

  return (
    <div className='w-full flex item-center justify-center'>
      <div className='text-xs lg:text-sm'>
      Facing Health issue? Try this to know more
      </div>
    </div>
  )

}


const Home = () => {
  return (
    // <div>
    //     jgusg


    // </div>
    <>
      {/* <Navbar>   </Navbar>
        <div className='p-10'>
            <img src={doctors} className='max-w-96 ' alt="" />

        </div> */}

      <Container>
        {/* <input type="text" /> */}

        <div className='flex flex-col  gap-5 justify-center'>

        <Detail/>
        <Selector className=""></Selector>


        {/* <div className='l'> */}
        {/* <RightPart ></RightPart> */}
        {/* </div> */}
        

        </div>
        {/* <Selector></Selector>
        <RightPart></RightPart> */}

      </Container>

        

    </>




  )
}


export default Home