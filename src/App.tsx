
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./components/Home/Home";
import Howitworks from "./components/How_it_works/Howitworks";
import Scanorupload from "./components/Scan_or_Upload/Scanorupload";
import Footer from "./components/Footer";
import Part5 from "./components/Part5";

const App = () => {
  return (
    <>
      <div className="overflow-x-hidden">
        <Navbar/>
         <Routes>
          <Route path="/" element={<Home/>} />
          <Route path="/howitworks" element={<Howitworks/>} />
          <Route path="/scanorupload" element={<Scanorupload />} />
        </Routes>
        <Part5/>
        <Footer/>
      </div>
    </>
  );
};

export default App;