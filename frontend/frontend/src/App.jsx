// add the beginning of your app entry
// import 'vite/modulepreload-polyfill'


// import { useState } from 'react'
// import '/static/index.css'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'
// import Navbar from './components/Navbar'

import React, { useState } from 'react'
import Navbar from './components/Navbar'
import { Routes, Route } from 'react-router-dom'
import Home from './components/Home'
import About from './components/About'
import ContactUs from './components/ContactUs'
import Disease from './components/Disease'
import ProtectedRoute from './ProtectedRoute/ProtectedRoute'
import { AccessProvider } from './AcessProvider/AcessProvider'
import { ChosenProvider } from './Context/ChosenContext'
import { PredictionProvider } from './Context/PredictionContext'

import { createContext } from 'react'

export const ThemeContext = createContext(null)

function App() {
  // const [count, setCount] = useState(0)
  const reactLogo = "/static/react.svg"
  const viteLogo = "/static/vite.svg"
  
  const [theme, setTheme] = useState("light");

  const toggleTheme =()=>{
    setTheme((curr)=>(curr === "light"? "dark": "light"));
  }

  return (
    <>
      {/* <div className='bg-teal-500'>wd</div> */}
      {/* <Navbar/> */}

    <ThemeContext.Provider value={{theme, setTheme}}>
    <PredictionProvider> 
    <ChosenProvider>
    <Routes>
   
    <Route path='/' element={ <Home/>}>   </Route > 
    <Route path='/about' element={<About/>} ></Route>
    <Route path='/contactus' element ={<ContactUs/>} ></Route>
    <Route element={<ProtectedRoute />}>
            <Route path="/disease" element={<Disease />} />
      </Route>
    </Routes> 
    
    </ChosenProvider>

    </PredictionProvider>
    
    </ThemeContext.Provider>
      
    </>
  )
}

export default App
