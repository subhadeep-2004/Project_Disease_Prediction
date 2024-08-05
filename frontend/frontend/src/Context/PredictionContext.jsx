import React, { createContext, useState ,useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAccess } from '../AcessProvider/AcessProvider';
export const PredictionContext = createContext();

export const PredictionProvider = ({ children }) => {
  const [prediction, setPrediction] = useState(null);
  const[description, setDescription] = useState(null);
  const [ precaution, setPrecaution] = useState([]);
  const navigate = useNavigate();
  const { setHasAccess } = useAccess();

  return (
    <PredictionContext.Provider value={{ prediction, setPrediction,description,setDescription, precaution, setPrecaution }}>
      {children}
    </PredictionContext.Provider>
  );
};
