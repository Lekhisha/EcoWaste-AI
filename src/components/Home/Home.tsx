
import Hero from './Hero'
import About from './About'
import About2 from './About2'
import About3 from "./About3";
import Uses from './Uses';
const Home = () => {
  return (
    <div className='overflow-x-hidden'>
        <Hero/>
         <About/>
        <About2/>
        <About3/>
        <Uses/>
      
    </div>
  )
}

export default Home
