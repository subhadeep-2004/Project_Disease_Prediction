import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';

const DiseasePrecautionCard = ({ image, name, precautions }) => {
  return (
    <Card sx={{ maxWidth: 345 }}>
      <CardMedia
        component="img"
        height="140"
        image={image}
        alt={name}
      />
      <CardContent>
        <Typography gutterBottom variant="h5" component="div">
          {name}
        </Typography>
        <List>
          {precautions.map((precaution, index) => (
            <ListItem key={index}>
              <ListItemText primary={precaution} />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default DiseasePrecautionCard;
